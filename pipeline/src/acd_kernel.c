/* Mean-parameterized EACD/WACD likelihood; all durations and scales supplied
   by the explicit chronological observation layer. No data reads or fitting. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

int acd_kernel(size_t n, const double *y, const uint8_t *event,
               const uint8_t *reset, double omega, double alpha, double beta,
               double shape, double log_gamma, double dlog_gamma,
               int weibull, int unit_init, double *objective, double *gradient,
               double *psi_out, double *hazard_out) {
    if(n==0 || omega<=0 || alpha<0 || beta<0 || alpha+beta>=1 || shape<=0) return 1;
    double p=0, dp[3]={0,0,0}, total=0, g[4]={0,0,0,0};
    for(size_t i=0;i<n;i++) {
        if(i==0 || reset[i]) {
            double q=1-alpha-beta;
            p=unit_init ? 1.0 : omega/q;
            dp[0]=unit_init ? 0.0 : 1/q;
            dp[1]=dp[2]=unit_init ? 0.0 : omega/(q*q);
        } else {
            if(!event[i-1]) return 2;
            double prev=p;
            p=omega+alpha*y[i-1]+beta*prev;
            dp[0]=1+beta*dp[0];dp[1]=y[i-1]+beta*dp[1];dp[2]=prev+beta*dp[2];
        }
        if(!isfinite(p) || p<=0 || !isfinite(y[i]) || y[i]<0 || (event[i] && y[i]<=0)) return 3;
        double h=0, loss=0, gp=0;
        if(weibull) {
            if(y[i]>0) {
                double z=log(y[i])+log_gamma-log(p);
                h=exp(shape*z);
                loss=h+event[i]*(log(p)-log_gamma-log(shape)-(shape-1)*z);
                gp=shape*(event[i]-h)/p;
                g[3]+=h*(z+shape*dlog_gamma)-event[i]*(z+shape*dlog_gamma+1/shape);
            }
        } else {
            h=y[i]/p;loss=h+event[i]*log(p);gp=(event[i]-h)/p;
        }
        if(!isfinite(loss)||!isfinite(h)) return 4;
        total+=loss;for(int j=0;j<3;j++) g[j]+=gp*dp[j];
        if(psi_out) psi_out[i]=p;
        if(hazard_out) hazard_out[i]=h;
    }
    *objective=total/n;for(int j=0;j<4;j++) gradient[j]=g[j]/n;
    return 0;
}
