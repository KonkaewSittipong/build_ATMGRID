"""Build ALL ATLAS bands (c + o) in ONE run.

SYNTHE computes the spectra once over the UNION wavelength range (min..max of all
filters, here ~4060-8320 A), and Stage C band-integrates each filter from the same
spectra -> one grid per filter. Adding more filters costs only their band integration.

    mpirun -n N python src/run_grid.py config_atlas.py
    python        src/run_grid.py config_atlas.py --assemble-only
"""
BASE   = "/lustre/MSSP/sittipong/buildmodule/build_ATMGRID"
MODELS = "spectro_grid/models"            # shared ATLAS12 models
H5DIR  = "spectro_grid/h5_atlas"          # shared spectra over the UNION range

# one dict per filter: file, name, desc, ext, out
FILTERS = [
    dict(file="filters/Misc_Atlas.c.txt", name="c", desc="ATLAS c", ext=1.228,
         out="spectro_grid/ATLAS12.AB.Atlas.c.h5"),
    dict(file="filters/Misc_Atlas.o.txt", name="o", desc="ATLAS o", ext=0.846,
         out="spectro_grid/ATLAS12.AB.Atlas.o.h5"),
]

TEFF = [3500, 13000, 250]
LOGG = [2.5, 5.5, 0.5]
FEH  = [0.0]

# the union range includes the line-dense c blue (~4100 A), so use low R
# (R-converged for broadband photometry; keeps the hot-model Balmer cost down)
RESOLU = 5000

# mu axis: stock icarus 91 (interp from native 17; see config_o.py for MU_NATIVE)
from mu_icarus import MU_ICARUS as MU
MU_NATIVE = False
