"""Config for the ATLAS c-band grid.  Run:  mpirun -n N python run_grid.py config_c.py

Reuses the same ATLAS12 models as o; only SYNTHE (over the c range) + assembly run.
The c band reaches into the line-dense blue (~4100 A), so it uses lower R (still
R-converged for broadband photometry) and node-local scratch (handled by run_grid.py)
to stay fast.
"""
BASE    = "/lustre/MSSP/sittipong/buildmodule/build_ATMGRID"
MODELS  = "spectro_grid/models"                    # SHARED models (same atmospheres as o)
H5DIR   = "spectro_grid/h5_c"                       # SEPARATE from o (different lambda range)
OUTGRID = "spectro_grid/ATLAS12.AB.Atlas.c.h5"

FILTER      = "filters/Misc_Atlas.c.txt"
FILTER_NAME = "c"
FILTER_DESC = "ATLAS c"
EXT         = 1.228                                  # Schlafly-scale ext (pivot ~5368 A)

# match the mu axis of the stock icarus grids (91 angles) instead of SYNTHE's native 17
from mu_icarus import MU_ICARUS as MU
MU_NATIVE = False   # True = compute all 91 mu natively (exact, ~8x cost); False = interp from 17

TEFF = [3500, 13000, 250]
LOGG = [2.5, 5.5, 0.5]
FEH  = [0.0]

RESOLU = 5000                  # low R for the line-dense blue: keeps the expensive
                               # Balmer-line computation in hot models cheap. R=5e3 is
                               # R-converged for broadband photometry (the band integral
                               # depends on total line blanketing, not line shapes).
