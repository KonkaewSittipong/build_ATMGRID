"""Small test config (21 cool dwarfs).  Run:  mpirun -n 14 python run_grid.py config_test.py
Writes to its own test_grid/ so it never touches the real grid."""
BASE    = "/lustre/MSSP/sittipong/buildmodule/build_ATMGRID"
MODELS  = "test_grid/models"
H5DIR   = "test_grid/h5"
OUTGRID = "test_grid/ATLAS12.AB.Atlas.o.h5"

FILTER      = "filters/Misc_Atlas.o.txt"
FILTER_NAME = "o"
FILTER_DESC = "ATLAS o"
EXT         = 0.846

TEFF = [3500, 5000, 250]       # 7 points
LOGG = [4.0, 5.0, 0.5]         # 3 points
FEH  = [0.0]

RESOLU = 20000
