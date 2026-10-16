"""Auditable acquisition entry point.

Use python download_dukascopy.py audit, then fetch, then build.
The legacy implementation is archived under reports/repair_history/.
No monthly file is treated as proof of complete coverage.
"""
from repair_acquisition import main

if __name__ == '__main__':
    main()
