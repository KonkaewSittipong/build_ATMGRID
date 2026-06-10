"""Config for the ATLAS o-band grid.  Run:  mpirun -n N python run_grid.py config_o.py"""

# paths (relative paths are resolved against BASE)
BASE    = "/lustre/MSSP/sittipong/buildmodule/build_ATMGRID"
MODELS  = "spectro_grid/models"                    # ATLAS12 models (shared across bands)
H5DIR   = "spectro_grid/h5"                         # node spectra (per band — see note)
OUTGRID = "spectro_grid/ATLAS12.AB.Atlas.o.h5"       # output photometry grid

# filter
FILTER      = "filters/Misc_Atlas.o.txt"            # 2-col: wavelength_A  transmission
FILTER_NAME = "o"
FILTER_DESC = "ATLAS o"
EXT         = 0.846                                  # Schlafly-scale ext (pivot ~6827 A)

# mu axis: match the stock icarus grids (91 angles) instead of SYNTHE's native 17
from mu_icarus import MU_ICARUS as MU
# MU_NATIVE=False (default): SYNTHE computes native 17, Stage C interpolates to MU
#   (the 91). Agrees with native-91 to ~1 mmag mean (≤8 mmag only at the extreme
#   limb, which barely contributes flux) — and ~8x cheaper.
# MU_NATIVE=True: SYNTHE computes all of MU directly in card-safe batches (exact,
#   no interpolation, ~8x the SYNTHE cost).
MU_NATIVE = False

# grid axes:  [start, stop, step]  (inclusive)
TEFF = [3500, 13000, 250]      # K
LOGG = [2.5, 5.5, 0.5]         # cgs log g
FEH  = [0.0]                   # list of [M/H]

# SYNTHE
RESOLU = 100000                # resolving power

# Optional (defaults in run_grid.py): VTURB=2.0, ITER=45, ATLAS_MOLECULES=False,
# ABUND=None, CLEAN_ATLAS=True, FALLBACK_ROUNDS=2, FALLBACK_NEIGHBOURS=4,
# WMARGIN=50.0, SYNTHE_MOLECULES=False, FILTER_CONV=1.0
#
# NOTE: H5DIR must be SEPARATE per band — node spectra cover the filter's
# wavelength range, so o and c spectra differ and must not share a directory.
