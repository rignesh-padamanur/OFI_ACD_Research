"""Censor-aware working ACD likelihood; chronological reference audit, not inference."""
from pathlib import Path
import ctypes
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, digamma

CEILING=.999
HAZARD_EDGES=np.r_[0.,-np.log1p(-np.array([.01,.05,.1,.5,.9,.99])),np.inf]
_KERNEL=None


def kernel_build_command(source, binary):
    """Build the unchanged C kernel with a local compiler on macOS or Linux.

    The default macOS command retains the archived build flags. Linux uses a
    position-independent shared object. CC may name a compiler with arguments;
    it is parsed as an argument vector and is never executed through a shell.
    """
    if sys.platform not in ('darwin', 'linux'):
        raise RuntimeError('The ACD kernel supports macOS and Linux. On Windows, use WSL with a Linux C compiler.')
    requested = os.environ.get('CC')
    compiler = shlex.split(requested) if requested else ['/usr/bin/clang' if sys.platform == 'darwin' else 'cc']
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError('A C99 compiler is required for the ACD kernel; install clang or GCC and set CC if needed.')
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    command = compiler + flags + ['-O3', '-std=c99', '-ffp-contract=off', str(source), '-o', str(binary)]
    if sys.platform == 'linux':
        command.append('-lm')
    return command, compiler


def get_kernel():
    global _KERNEL
    if _KERNEL is not None:return _KERNEL
    source=Path(__file__).with_name('acd_kernel.c');fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
    directory=Path(tempfile.gettempdir())/'fx_acd_kernel'/fingerprint;directory.mkdir(parents=True,exist_ok=True)
    binary=directory/('kernel.dylib' if sys.platform == 'darwin' else 'kernel.so')
    command,compiler=kernel_build_command(source,binary)
    # Build from reviewed local source every interpreter run; do not trust a cached binary.
    subprocess.run(command,check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str(binary));f=lib.acd_kernel
    double=ctypes.POINTER(ctypes.c_double);byte=ctypes.POINTER(ctypes.c_uint8)
    f.argtypes=[ctypes.c_size_t,double,byte,byte]+[ctypes.c_double]*6+[ctypes.c_int]*2+[double]*4
    f.restype=ctypes.c_int
    info={'source_sha256':fingerprint,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
          'command':command,'compiler':subprocess.run(compiler+['--version'],capture_output=True,text=True,check=True).stdout}
    _KERNEL=f,info
    return _KERNEL


def arrays(frame):
    y=np.ascontiguousarray(frame.normalized_exposure,dtype='float64')
    event=np.ascontiguousarray(frame.event_observed,dtype='uint8');reset=np.ascontiguousarray(frame.recurrence_reset,dtype='uint8')
    if not len(y) or not np.isfinite(y).all() or (y<0).any() or ((y<=0)&(event!=0)).any():raise ValueError('Invalid likelihood exposures')
    if not reset[0] or ((~event[:-1].astype(bool))&(~reset[1:].astype(bool))).any():raise ValueError('State continued after an incomplete wait')
    return y,event,reset


def evaluate(data,params,*,weibull=False,unit_init=False,predictions=False):
    y,event,reset=data;omega,alpha,beta,shape=params
    if not np.isfinite(params).all() or omega<=0 or min(alpha,beta)<0 or alpha+beta>=1 or shape<=0:
        raise ValueError('Invalid ACD parameters')
    if any(len(a)!=len(y) for a in [event,reset]):raise ValueError('Inconsistent likelihood arrays')
    objective=np.zeros(1);gradient=np.zeros(4)
    psi=np.empty(len(y)) if predictions else None;hazard=np.empty(len(y)) if predictions else None
    dp=ctypes.POINTER(ctypes.c_double);bp=ctypes.POINTER(ctypes.c_uint8)
    code=get_kernel()[0](len(y),y.ctypes.data_as(dp),event.ctypes.data_as(bp),reset.ctypes.data_as(bp),
        omega,alpha,beta,shape,float(gammaln(1+1/shape)),float(-digamma(1+1/shape)/shape**2),
        int(weibull),int(unit_init),objective.ctypes.data_as(dp),gradient.ctypes.data_as(dp),
        psi.ctypes.data_as(dp) if predictions else None,hazard.ctypes.data_as(dp) if predictions else None)
    if code:raise ValueError('Invalid ACD recurrence or likelihood, kernel code '+str(code))
    return float(objective[0]),gradient,psi,hazard


def transform(theta,weibull):
    logits=np.r_[0.,theta[1:3]];weights=np.exp(logits-logits.max());weights/=weights.sum()
    omega=float(np.exp(theta[0]));alpha,beta=CEILING*weights[1:]
    return np.array([omega,alpha,beta,float(np.exp(theta[3])) if weibull else 1.])


def transformed_objective(theta,data,*,weibull,unit_init=False):
    params=transform(theta,weibull);value,g,_,_=evaluate(data,params,weibull=weibull,unit_init=unit_init)
    w,a,b,k=params
    jac=[g[0]*w,g[1]*a*(1-a/CEILING)-g[2]*a*b/CEILING,
         g[2]*b*(1-b/CEILING)-g[1]*a*b/CEILING]
    if weibull:jac.append(g[3]*k)
    return value,np.asarray(jac)


def fit_acd(frame,*,weibull=False,unit_init=False):
    data=arrays(frame);y,event,_=data
    if event.sum()<100:raise ValueError('Insufficient completed waits for reference fit')
    mean=float(y.sum()/event.sum());attempts=[];best=None
    bounds=[(-12,6),(-12,12),(-12,12)]+([(np.log(.2),np.log(5))] if weibull else [])
    for a,b,k in [(.05,.85,.7),(.15,.70,1.),(.30,.50,.5)]:
        theta=np.array([np.log(mean*(1-a-b)),np.log(a/(CEILING-a-b)),np.log(b/(CEILING-a-b))]+([np.log(k)] if weibull else []))
        result=minimize(lambda th:transformed_objective(th,data,weibull=weibull,unit_init=unit_init),
            theta,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-11,'gtol':1e-6,'maxls':30})
        p=transform(result.x,weibull);record={'start_alpha':a,'start_beta':b,'start_shape':k if weibull else 1.,
            'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),
            'objective':float(result.fun),'gradient_inf_norm':float(np.max(np.abs(result.jac))),
            'omega':float(p[0]),'alpha':float(p[1]),'beta':float(p[2]),'shape':float(p[3]),
            'near_constraint':bool(p[1]+p[2]>CEILING-1e-5 or (weibull and (p[3]<.2001 or p[3]>4.999)))}
        attempts.append(record)
        if result.success and np.isfinite(result.fun) and (best is None or result.fun<best['objective']):best=record
    if best is None:return None,attempts
    return best,attempts


def predict(frame,params,*,weibull=False,unit_init=False):
    value,grad,psi,h=evaluate(arrays(frame),params,weibull=weibull,unit_init=unit_init,predictions=True)
    y=frame.normalized_exposure.to_numpy();event=frame.event_observed.to_numpy();scale=frame.scale_sec.to_numpy()
    if weibull:
        k=params[3];lg=gammaln(1+1/k)
        logy=np.zeros(len(y));np.log(y,out=logy,where=y>0)
        nll=h+event*(np.log(psi)-lg-np.log(k)-(k-1)*(logy+lg-np.log(psi)))
    else:nll=h+event*np.log(psi)
    raw=nll+event*np.log(scale)
    return pd.DataFrame({'origin_quote_id':frame.origin_quote_id.to_numpy(),'psi':psi,'cumulative_hazard':h,
        'negative_log_score_seconds':raw,'martingale_residual':event.astype('float64')-h})


def diagnostics(frame,predictions):
    """Descriptive working-model diagnostics; no independence-based p-values."""
    h=predictions.cumulative_hazard.to_numpy();event=frame.event_observed.to_numpy()
    mart=event-h;reset=frame.recurrence_reset.to_numpy();eligible_pairs=~reset[1:]
    corr=float(np.corrcoef(mart[:-1][eligible_pairs],mart[1:][eligible_pairs])[0,1]) if eligible_pairs.sum()>1 else None
    if corr is not None and not np.isfinite(corr):corr=None
    ratios=[];bands=[]
    for lo,hi in zip(HAZARD_EDGES[:-1],HAZARD_EDGES[1:],strict=True):
        exposure=float(np.maximum(np.minimum(h,hi)-lo,0).sum());n=int((event&(h>=lo)&(h<hi)).sum())
        ratio=n/exposure if exposure>0 else None
        bands.append({'lower_hazard':float(lo),'upper_hazard':float(hi) if np.isfinite(hi) else None,
            'events':n,'hazard_exposure':exposure,'event_exposure_ratio':ratio})
        if exposure>=100 and ratio is not None:ratios.append(ratio)
    total_ratio=float(event.sum()/h.sum()) if h.sum()>0 else None
    flags=[]
    if total_ratio is None or not .9<=total_ratio<=1.1:flags.append('total_hazard_calibration')
    if corr is None or not np.isfinite(corr) or abs(corr)>.05:flags.append('martingale_lag1_dependence')
    if any(not .8<=r<=1.2 for r in ratios):flags.append('hazard_band_calibration')
    return {'spells':len(frame),'completed_events':int(event.sum()),'incomplete_positive_exposures':int((~event).sum()),
        'sequence_resets':int(reset.sum()),'sequence_fragments':int(1+reset[1:].sum()),
        'negative_log_score_seconds_mean':float(predictions.negative_log_score_seconds.mean()),
        'total_hazard':float(h.sum()),'event_hazard_ratio':total_ratio,'martingale_mean':float(mart.mean()),
        'martingale_lag1_correlation':corr,'adjacent_pairs':int(eligible_pairs.sum()),'hazard_bands':bands,
        'diagnostic_flags':flags,'reference_diagnostic_screen_passed':not flags,
        'inference':'Descriptive predeclared screening tolerances, conditional on working observation assumptions; no formal p-values.'}


def daily_diagnostics(frame,predictions):
    f=pd.DataFrame({'date':pd.DatetimeIndex(frame.origin_at).normalize(),'event':frame.event_observed.to_numpy().astype('int64'),
        'hazard':predictions.cumulative_hazard.to_numpy(),'negative_log_score_seconds':predictions.negative_log_score_seconds.to_numpy(),
        'martingale_residual':predictions.martingale_residual.to_numpy()})
    return f.groupby('date',as_index=False).agg(spells=('event','size'),completed_events=('event','sum'),
        total_hazard=('hazard','sum'),mean_negative_log_score=('negative_log_score_seconds','mean'),martingale_mean=('martingale_residual','mean'))
