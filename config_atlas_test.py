"""SMALL multi-band validation: build c + o in ONE run on a tiny 2x2 grid.

Proves the multi-filter path (SYNTHE once over the union range, Stage C band-
integrates each filter -> one grid per filter). Writes to a separate test dir so it
does NOT touch the production spectro_grid/ATLAS12.AB.Atlas.{c,o}.h5.

    mpirun -n N python src/run_grid.py config_atlas_test.py
    python        src/run_grid.py config_atlas_test.py --assemble-only
"""
BASE   = "/lustre/MSSP/sittipong/buildmodule/build_ATMGRID"
MODELS = "spectro_grid_test/models"
H5DIR  = "spectro_grid_test/h5_atlas"

FILTERS = [
    dict(file="filters/Misc_Atlas.c.txt", name="c", desc="ATLAS c", ext=1.228,
         out="spectro_grid_test/ATLAS12.AB.Atlas.c.h5"),
    dict(file="filters/Misc_Atlas.o.txt", name="o", desc="ATLAS o", ext=0.846,
         out="spectro_grid_test/ATLAS12.AB.Atlas.o.h5"),
]

# tiny, mid-range (fast, well-converging) grid: Teff {6000,7000}, logg {4.5,5.0}
TEFF = [6000, 7000, 1000]
LOGG = [4.5, 5.0, 0.5]
FEH  = [0.0]

RESOLU = 5000

from mu_icarus import MU_ICARUS as MU
MU_NATIVE = False
