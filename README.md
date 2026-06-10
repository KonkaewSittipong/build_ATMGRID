# build_ATMGRID — ATLAS12 specific-intensity photometry grids for Icarus

Builds Icarus atmosphere photometry grids (`ATLAS12.AB.<inst>.<band>.h5`) from
ATLAS12 model atmospheres + SYNTHE spectral synthesis. For a grid of
(temperature × gravity) stars it computes each star's emergent spectrum at many
viewing angles, then band-integrates through a filter into one brightness number
per angle. The output is byte-compatible with the stock Icarus grids in
`packages/icarus_packages/atmo/`.

Runs with the **hcam-env** conda python (`/home/sittipong/.conda/envs/hcam-env/bin/python`,
3.12). All driver code lives in `src/` — including its own copies of `run_atlas12.py`
and `synthe.py` (their `HERE` repointed to the `packages/ATLAS` install, which still
holds the ATLAS12/SYNTHE executables and line lists). Icarus itself is not imported
(its compiled extensions are py3.10 only); the HDF5 format and band-integration math
are replicated directly with h5py/scipy.

## Quick start

```bash
sbatch submit_run_grid.sh
```

`run_grid.py` is the **pipeline**; the band/grid parameters live in a **config
file** passed as an argument. So:

```bash
mpirun -n N python src/run_grid.py config_c.py            # Stages A+B (parallel)
python        src/run_grid.py config_c.py --assemble-only # Stage C (serial, durable)
```

The submit scripts wrap exactly that (`submit_run_grid.sh` -> `config_o.py`,
`submit_run_grid_c.sh` -> `config_c.py`, `submit_run_grid_test.sh` -> `config_test.py`).
Resume-safe: re-running fills only what is missing.

## Pipeline (run_grid.py + a config file)

```
 config_<band>.py: paths, TEFF/LOGG=[start,stop,step], filter, ext, RESOLU
        |
  Stage A  ATLAS12 models        build_model() -> run_atlas12.run_phases() [in-process/rank]
           fallback: a model that diverges (NaN) is retried from the nearest
           CONVERGED neighbours (A/F instability band ~6500-8250 K)
        |
  Stage B  SYNTHE surface I(mu)   build_spectro_grid.build_intensity_bin() -> spectrum.bin
           write_spectro_h5() -> <H5DIR>/<node>.spectro.h5   (+ fsync)
           runs on node-local scratch (/dev/shm) so the big fort.* tapes don't
           thrash lustre when many ranks run at once
        |
  Stage C  band-integrate         assemble(): each spectrum x filter -> flux(nT,ng,nmu)
        |
        v
  spectro_grid/ATLAS12.AB.<band>.h5      <- the product
```

`<node>` = `run_t<Teff>g<logg*10><p|m><|feh|*10>`, e.g. `run_t5500g40p00`.

## Files

### Layout
```
build_ATMGRID/
├── README.md
├── config_o.py, config_c.py, config_atlas.py, config_test.py   # params (edit these)
│       (config_atlas.py = multi-filter: c+o in one run; others single-band)
├── submit_run_grid*.sh                         # SLURM launchers (sbatch these)
├── src/                                        # all driver code (self-contained)
│   ├── run_grid.py            # the pipeline (loads a config, runs all 3 stages)
│   ├── build_spectro_grid.py  # node worker: model -> SYNTHE I(mu) -> spectro h5
│   ├── band_integrate.py      # filter loading + AB band-integration / pivot
│   ├── assemble_grid.py       # standalone Stage-C assembler
│   ├── run_atlas12.py         # ATLAS12 driver (own copy; exes/data in packages/ATLAS)
│   ├── synthe.py              # SYNTHE driver  (own copy; exes/data in packages/ATLAS)
│   └── mu_icarus.py           # the 91-angle icarus mu axis (MU_ICARUS)
├── filters/Misc_Atlas.{o,c}.txt   # filter curves (2 cols: wavelength_A transmission)
├── legacy/                        # superseded scripts (kept for reference)
└── spectro_grid/                  # outputs: models/ (shared), h5/ & h5_c/, final grids
```
Config and launchers are top-level (the things you edit/run); all code is in `src/`.
The submit scripts `cd` to this dir and call `python src/run_grid.py <config>`.

## Configuration (a config_<band>.py file)

```python
BASE    = "/lustre/.../build_ATMGRID"
MODELS  = "spectro_grid/models"          # ATLAS12 models (SHARED across bands)
H5DIR   = "spectro_grid/h5_c"            # node spectra — SEPARATE per band
OUTGRID = "spectro_grid/ATLAS12.AB.Atlas.c.h5"
FILTER  = "filters/Misc_Atlas.c.txt"
FILTER_NAME = "c"; FILTER_DESC = "ATLAS c"; EXT = 1.228

TEFF = [3500, 13000, 250]   # [start, stop, step] inclusive (K)
LOGG = [2.5, 5.5, 0.5]      # [start, stop, step] inclusive (cgs)
FEH  = [0.0]                # list of [M/H]
RESOLU = 20000              # SYNTHE resolving power
# Optional keys (defaults in run_grid.py _DEFAULTS): VTURB, ITER, ATLAS_MOLECULES,
# ABUND, CLEAN_ATLAS, FALLBACK_ROUNDS, FALLBACK_NEIGHBOURS, WMARGIN,
# SYNTHE_MOLECULES, FILTER_CONV, SCRATCH (=/dev/shm).
# WSTART/WEND are read from the filter .txt span (+/- WMARGIN) automatically.
```

### mu axis (native 17 vs icarus 91)
SYNTHE computes intensity at ≤20 angles (the `ANGLE(20)` limit); the stock icarus
grids use 91. Two ways to land on the 91-mu axis, set per config:
- **`MU_NATIVE = False`** (default): SYNTHE computes the native 17, Stage C
  **interpolates** the band flux onto `MU` (exact, since band-integration is linear
  in intensity). Measured vs native-91: **~1 mmag mean**, ≤8 mmag only at the
  extreme limb (negligible flux). ~8× cheaper.
- **`MU_NATIVE = True`**: SYNTHE computes all of `MU` **directly**, in card-safe
  batches of `MU_BATCH` (=12) angles, merged — exact, no interpolation, ~8× the cost.

Set the axis with `from mu_icarus import MU_ICARUS as MU` in the config.

### Multiple filters in one run (recommended)
Give a config a **`FILTERS`** list and the pipeline computes the spectra **once**
over the **union** wavelength range, then band-integrates **every** filter from the
same spectra — one grid per filter (see `config_atlas.py`, builds c+o together):
```python
H5DIR   = "spectro_grid/h5_atlas"           # shared spectra over the union range
FILTERS = [
    dict(file="filters/Misc_Atlas.c.txt", name="c", desc="ATLAS c", ext=1.228,
         out="spectro_grid/ATLAS12.AB.Atlas.c.h5"),
    dict(file="filters/Misc_Atlas.o.txt", name="o", desc="ATLAS o", ext=0.846,
         out="spectro_grid/ATLAS12.AB.Atlas.o.h5"),
]
RESOLU = 5000   # union includes the c blue -> low R (R-converged for broadband)
```
Adding a filter costs only its band-integration, not another SYNTHE pass.

### Build a single band
A config may instead use the single-band form (`FILTER`/`FILTER_NAME`/`FILTER_DESC`/
`EXT`/`OUTGRID`) — see `config_o.py`/`config_c.py`. Use a **separate `H5DIR` per band**
(spectra cover only that filter's range). Models are reused — no ATLAS12 re-run.
The ATLAS12 models are reused; only the new SYNTHE range + assembly run.
`ext` (Icarus convention = `A_band/A_V / 0.78`) can be interpolated from the
existing `atmo/ATLAS9.AB.*.h5` grids' `ext`-vs-`pivot` values.

## Output format (Icarus AtmoGridPhot)

```
flux  (n_logtemp, n_logg, n_mu)   log10 <f_nu> [dex(erg/(Angstrom cm2 s))]
cols/ logtemp, logg, mu
meta  Z, magsys=AB, zp=-48.6, filter, pivot, ext
```

## Design notes / gotchas

- **The blue is expensive.** `synthe.exe` cost scales with line count, and the
  4000-5400 A region (dense metal lines + the Balmer series for hot stars) is the
  bottleneck. At R=1e5 it can blow past a 12 h walltime; **drop `RESOLU` to ~5000**
  (R-converged for broadband — see "mu axis"/R notes) so the c band runs in a
  reasonable time. Even then, hot-model Balmer is the slow tail.
- **Use a dedicated node for blue-heavy bands.** `chalawan_cpu` (castor*) nodes are
  shared — if others load the node, your ranks get starved (e.g. 42% CPU) and the
  tail drags. Prefer a free node, and note the SYNTHE cost is uneven across the grid
  (hot models slowest), so the last few ranks can lag — fewer ranks / resume-fill
  the stragglers if needed.
- **Stage C must run OUTSIDE mpirun** (it does, via `--assemble-only`). Run inside
  mpirun, rank 0 writes+closes the grid but the job can exit before lustre syncs
  -> 0-byte file. Stage B spectra are protected by `fsync` for the same reason.
- **Don't run ATLAS12 on the login node** (1 core, shared) — use SLURM.
- **Node-local scratch:** SYNTHE runs in `/dev/shm` (config `SCRATCH`); it must be
  node-local, NOT `$TMPDIR` (which points to lustre here) or the big `fort.*` tapes
  thrash the filesystem when many ranks run at once.
- **Atomic lines only** by default — cool nodes (Teff < ~4500 K) miss TiO/molecular
  bands. Set `ATLAS_MOLECULES`/`SYNTHE_MOLECULES` for those (data in
  `packages/ATLAS/lines/molecules/`).
- **mu = 91 (icarus axis).** SYNTHE computes native 17 (max 20: `ANGLE(20)`); Stage C
  interpolates the band flux to the icarus 91 (`MU = MU_ICARUS`, sub-mmag vs native).
  `MU_NATIVE=True` computes all 91 directly in card-safe batches (exact, ~8x cost).

## Status (2026-06-10)

- **o band — done:** `spectro_grid/ATLAS12.AB.Atlas.o.h5` — **(41, 7, 91), 287/287,
  0 NaN**, pivot 6827 A, ext 0.846, AB. All 12 A/F convergence failures filled via
  the neighbour-start fallback; mu resampled to the icarus 91.
- **c band — in progress:** `ATLAS12.AB.Atlas.c.h5` at R=5000, full line list.
  Slow tail (hot-model Balmer + shared-node contention); resume-fill the stragglers.
- **Multi-filter:** `config_atlas.py` builds c+o together (spectra over the union
  range, band-integrated through both). Preferred for new bands.

TODO: finish c; molecules for cool nodes (≲4500 K); proper band-integrated `ext`
for c/o (currently interpolated from the library's ext-vs-pivot).
