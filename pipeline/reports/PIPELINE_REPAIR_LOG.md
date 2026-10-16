# Legacy execution guard

This release retains this path because `src/pipeline_state.py` uses its existence
to block superseded notebook and persistence APIs. It is an execution guard,
not the full historical repair log.

The public source tree contains selected current algorithms, their dependencies,
and tests. Historical `repair_*.py` runners also require their registered data,
manifests, frozen sources, and prior registration state. Do not run them as a
new raw-to-results build inside this compact publication package. Restore the
separate research archive and follow the reproduction instructions first.

The complete research archive retains the original `reports/PIPELINE_REPAIR_LOG.md`.
