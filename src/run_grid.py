#!/usr/bin/env python3
"""ATLAS12 -> SYNTHE surface-intensity -> photometry-grid pipeline.

The band/grid parameters live in a SEPARATE config file passed as an argument
(see config_o.py for the template). Usage:

    mpirun -n N python run_grid.py config_c.py        # Stages A+B (models + SYNTHE)
    python        run_grid.py config_c.py --assemble-only   # Stage C (serial, durable)

Stages (all in-process across the same MPI ranks — no nested mpirun):
  A  ATLAS12 models   build_model() -> run_atlas12.run_phases(); on divergence
                      (A/F instability band) retry from converged neighbours.
  B  SYNTHE I(mu)     build_spectro_grid.build_intensity_bin() -> spectro h5.
                      Runs on node-local scratch ($TMPDIR) to avoid lustre I/O
                      contention from the big fort.* line tapes; only the final
                      spectrum is written to lustre.
  C  band-integrate   assemble(): each spectrum x filter -> flux(nT,ng,nmu).
                      MUST run serially (--assemble-only), not inside mpirun.

Resume-safe: existing models / node spectra are skipped.
See DOCUMENT.md for the physics/math behind each step.
"""
# ── standard-library imports ────────────────────────────────────────────────
import importlib.util            # to load a config .py given by path at runtime
import os                        # os.fsync (durable writes), path checks
import re                        # parse node-dir names (run_tXXXXgYYpZZ)
import shutil                    # copy start models, remove scratch dirs
import sys                       # argv, sys.path, sys.exit
import tempfile                  # node-local scratch dirs (mkdtemp)
from itertools import product    # build the Teff x logg x feh grid
from pathlib import Path         # all filesystem paths
from types import SimpleNamespace  # build the `args` object run_phases expects

# ── third-party imports ─────────────────────────────────────────────────────
import numpy as np               # arrays, interpolation, integration helpers
import h5py                      # read node spectra / write photometry grids
from astropy.time import Time    # timestamp for the grid 'description' attr

# ── make the sibling modules in this src/ dir importable ────────────────────
HERE = Path(__file__).resolve().parent            # src/ — all driver modules live here
sys.path.insert(0, str(HERE))                     # so `import run_atlas12` etc. find them
import run_atlas12 as A           # ATLAS12 worker functions   (src/run_atlas12.py)
import build_spectro_grid as B    # SYNTHE intensity + spectro-h5 writer (src/)
import band_integrate as BI       # filter loading / AB band integration / pivot

# ── config defaults ──────────────────────────────────────────────────────────
# Every config key not set in the config file falls back to one of these.
# (Required keys with no default: MODELS, H5DIR, TEFF, LOGG, and a filter spec.)
_DEFAULTS = dict(
    FEH=[0.0],                   # list of [M/H] values for the grid
    FILTER_CONV=1.0,             # multiply filter col-1 by this to get Angstrom
    WMARGIN=50.0,                # Angstrom padding added around the filters' span
    RESOLU=100000,               # SYNTHE resolving power R = lambda/dlambda
    VTURB=2.0,                   # ATLAS12 microturbulence (km/s)
    ITER=45,                     # ATLAS12 phase-2 iterations
    ATLAS_MOLECULES=False,       # molecular equilibrium in the ATLAS12 model
    ABUND=None,                  # abundance-override file for ATLAS12 (or None)
    CLEAN_ATLAS=True,            # delete ATLAS12 intermediates after each model
    FALLBACK_ROUNDS=2,           # passes of neighbour-start retry for failed models
    FALLBACK_NEIGHBOURS=4,       # nearest converged neighbours to try per failure
    SYNTHE_MOLECULES=False,      # molecular line lists in SYNTHE
    # node-local scratch for SYNTHE fort.* tapes. MUST be node-local (NOT lustre /
    # $TMPDIR) or the big tapes thrash the filesystem when many ranks run at once.
    SCRATCH="/dev/shm",          # RAM tmpfs = fastest node-local scratch
    # mu axis for the output grid. Stage C interpolates the band flux from SYNTHE's
    # native 17 mu onto this axis (exact: band integration is linear in intensity).
    # Give MU as an explicit list (e.g. `from mu_icarus import MU_ICARUS as MU`) or
    # MU_REF as a grid path whose mu to copy. None/None = keep the native 17.
    MU=None, MU_REF=None,
    # MU_NATIVE=True: SYNTHE computes the MU angles DIRECTLY in card-safe batches of
    # MU_BATCH (exact, ~Nbatch x slower). False (default): native 17 + interpolate.
    MU_NATIVE=False, MU_BATCH=12,
)


def load_config(path):
    """Load a config .py and populate this module's globals (MODELS, FILTERS, ...).

    A config may specify either a single band (FILTER/FILTER_NAME/FILTER_DESC/EXT/
    OUTGRID) or many (a FILTERS list). Both are normalised to a FILTERS list here,
    and WSTART/WEND are set to the union of all filter wavelength spans so one SYNTHE
    pass serves every filter.
    """
    # import the config file as a throwaway module named "gridconfig"
    spec = importlib.util.spec_from_file_location("gridconfig", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)                  # runs the config's top-level code
    g = globals()                                 # we write the config into THIS module's globals
    base = Path(getattr(cfg, "BASE", HERE)).resolve()   # project root for relative paths

    def _p(x):                                    # resolve a path relative to BASE (abs paths pass through)
        p = Path(x)
        return p if p.is_absolute() else base / p

    g["BASE"]   = base
    g["MODELS"] = _p(cfg.MODELS)                  # dir of ATLAS12 model_final.dat (shared across bands)
    g["H5DIR"]  = _p(cfg.H5DIR)                   # dir of per-node spectro h5 (one wavelength range)
    for k in ("TEFF", "LOGG"):                    # required grid axes (no default)
        g[k] = getattr(cfg, k)
    for k, dv in _DEFAULTS.items():               # everything else: config value or the default
        g[k] = getattr(cfg, k, dv)

    # Normalise to a FILTERS list of dicts {file, name, desc, ext, out}.
    if getattr(cfg, "FILTERS", None):             # multi-band config
        flts = []
        for fd in cfg.FILTERS:
            flts.append(dict(file=_p(fd["file"]), name=fd["name"],
                             desc=fd.get("desc", fd["name"]),
                             ext=fd.get("ext", float("nan")), out=_p(fd["out"])))
    else:                                         # single-band config -> one-element list
        flts = [dict(file=_p(cfg.FILTER), name=cfg.FILTER_NAME,
                     desc=getattr(cfg, "FILTER_DESC", cfg.FILTER_NAME),
                     ext=getattr(cfg, "EXT", float("nan")), out=_p(cfg.OUTGRID))]
    g["FILTERS"] = flts

    # SYNTHE wavelength range = union of all filter spans (+/- WMARGIN), so the one
    # set of spectra covers every filter. (col 0 of each filter file = wavelength.)
    lo, hi = +np.inf, -np.inf
    for fd in flts:
        w = np.loadtxt(fd["file"])[:, 0] * g["FILTER_CONV"]
        lo, hi = min(lo, w.min()), max(hi, w.max())
    g["WSTART"] = float(np.floor(lo - g["WMARGIN"]))
    g["WEND"]   = float(np.ceil(hi + g["WMARGIN"]))


def frange(start, stop, step):
    """Inclusive float range [start, start+step, ..., stop] (avoids fp drift)."""
    vals, n = [], 0
    while True:
        v = round(start + n * step, 10)           # round to kill float accumulation error
        if v > stop + step * 1e-9:                # inclusive of `stop`
            break
        vals.append(v)
        n += 1
    return vals


def node_name(teff, logg, feh):
    """Grid-point directory name, matching run_atlas12's convention.
    e.g. (5500, 4.0, 0.0) -> 'run_t5500g40p00'  (g=logg*10, p/m + |feh|*10)."""
    sign = "p" if feh >= 0 else "m"
    return f"run_t{int(teff)}g{int(logg * 10):02d}{sign}{int(abs(feh) * 10):02d}"


# ── Stage A: one ATLAS12 model in-process ───────────────────────────────────

# regex to parse Teff/logg/feh back out of a node-dir name (for the neighbour search)
_NODE_RE = re.compile(r"^run_t(\d+)g(\d+)([pm])(\d+)$")


def converged_neighbours(teff, logg, feh, exclude):
    """Converged models (same feh) sorted by distance to (teff, logg), excluding
    `exclude`. Used as start models when the Kurucz-grid start diverges (A/F
    instability band ~6500-8250 K: deep-layer radiative accel runs away -> pressure
    NaN -> no fort.7). The nearest neighbour is not always a good seed, so the caller
    tries several in order until one converges."""
    cands = []
    for d in MODELS.glob("run_t*"):               # scan all node dirs
        if "_step" in d.name or d.name == exclude:  # skip step-ladder dirs and self
            continue
        mf = d / "model_final.dat"
        if not (mf.exists() and mf.stat().st_size > 0):   # only CONVERGED models
            continue
        m = _NODE_RE.match(d.name)
        if not m:
            continue
        t = int(m.group(1)); g = int(m.group(2)) / 10.0   # parse Teff, logg
        f = (1 if m.group(3) == "p" else -1) * int(m.group(4)) / 10.0   # parse feh
        if abs(f - feh) > 1e-6:                   # same metallicity only
            continue
        # distance metric: normalise Teff by 250 K and logg by 0.5 dex (grid steps)
        dist = ((t - teff) / 250.0) ** 2 + ((g - logg) / 0.5) ** 2
        cands.append((dist, mf))
    cands.sort(key=lambda x: x[0])                # nearest first
    return [mf for _, mf in cands]


def build_model(teff, logg, feh, start_model=None):
    """Build one ATLAS12 model. If start_model is given, begin from it (copied to
    start_model.dat); otherwise extract the closest Kurucz odfnew grid model.
    Returns (status, model_path) with status in {skip, ok, warn, fail}."""
    rundir = MODELS / node_name(teff, logg, feh)
    final  = rundir / "model_final.dat"
    if final.exists() and final.stat().st_size > 0:   # resume-safe: already built
        return "skip", final
    rundir.mkdir(parents=True, exist_ok=True)
    start_dat = rundir / "start_model.dat"
    if start_model is not None:                   # fallback: start from a neighbour model
        shutil.copy2(start_model, start_dat)
    else:                                         # normal: nearest Kurucz odfnew grid model
        A.extract_starting_model(A.pick_grid(feh), teff, logg, start_dat)
    # run_phases reads these fields off the `args` object
    args = SimpleNamespace(feh=feh, vturb=VTURB, iter=ITER,
                           molecules=ATLAS_MOLECULES, abund=ABUND)
    status = "fail"
    try:
        _, out = A.run_phases(rundir, start_dat, teff, logg, args)   # phase 1 + phase 2
        ok, _ = A.convergence_ok(out)             # parse flux-error from the phase-2 log
        status = "ok" if ok else "warn"           # 'warn' = punched but flux error high
    except SystemExit:                            # run_phases exits(1) if fort.7 not punched
        status = "fail"
    if CLEAN_ATLAS:                               # drop selected_lines.bin / logs (~190 MB)
        A._cleanup_intermediates(rundir)
    if not (final.exists() and final.stat().st_size > 0):   # final truth: did we get a model?
        return "fail", final
    return status, final


# ── Stage B: SYNTHE surface intensity for one model ─────────────────────────

def _scratch_base():
    """First writable node-local scratch dir (NOT lustre/$TMPDIR), for SYNTHE tapes."""
    for d in (SCRATCH, "/tmp"):                   # prefer /dev/shm (RAM), then /tmp (local disk)
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    return None                                   # fall back to tempfile default (last resort)


def build_spectrum(mp, out):
    """Run SYNTHE for model `mp` on node-local scratch, write spectro h5 `out`.
    Keeping the big fort.* tapes off lustre avoids the I/O contention that makes
    blue-heavy ranges crawl when many ranks run at once."""
    if MU_NATIVE:                                 # exact-91 mode: dispatch to the batched path
        build_spectrum_native(mp, out)
        return
    # default mode: compute the native 17 angles, then Stage C interpolates to MU
    scratch = Path(tempfile.mkdtemp(prefix=f"synthe_{mp.parent.name}_", dir=_scratch_base()))
    try:
        # run the 6-step SYNTHE chain -> spectrum.bin (mu-resolved intensity)
        binp = B.build_intensity_bin(scratch, mp, WSTART, WEND, RESOLU,
                                     0.0, SYNTHE_MOLECULES, [B.S.DEFAULT_LINELIST], "AIR")
        B.write_spectro_h5(binp, out)             # decode -> log Iλ -> HDF5 (+fsync)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)   # always clean scratch (RAM)


def build_spectrum_native(mp, out):
    """Compute the spectrum at ALL MU angles natively (no interpolation): SYNTHE in
    card-safe batches of MU_BATCH angles, merged into one spectro h5.
    ~ceil(len(MU)/MU_BATCH) x the SYNTHE cost of the interp path."""
    angles  = [float(a) for a in MU]              # the target angles (e.g. the icarus 91)
    # split into chunks small enough that the SURFACE INTENSI card fits ~80 cols
    batches = [angles[i:i + MU_BATCH] for i in range(0, len(angles), MU_BATCH)]
    scratch, bins = [], []
    try:
        for bi, ba in enumerate(batches):         # one SYNTHE run per angle-batch
            sc = Path(tempfile.mkdtemp(prefix=f"synN_{mp.parent.name}_{bi}_", dir=_scratch_base()))
            scratch.append(sc)
            bins.append(B.build_intensity_bin(sc, mp, WSTART, WEND, RESOLU, 0.0,
                                              SYNTHE_MOLECULES, [B.S.DEFAULT_LINELIST],
                                              "AIR", mu_angles=ba))
        B.write_spectro_h5_merged(bins, out)      # stitch the batches' mu together
    finally:
        for sc in scratch:                        # clean all batch scratch dirs
            shutil.rmtree(sc, ignore_errors=True)


# ── Stage C: assemble node spectra through the filter(s) (serial, durable) ───

def assemble():
    """Band-integrate the node spectra through EVERY filter in FILTERS and write one
    photometry grid per filter. Each spectrum is read once and integrated through all
    filters (the spectra span the union wavelength range), so adding filters is cheap."""
    files = sorted(H5DIR.glob("*.spectro.h5"))    # all per-node spectra
    if not files:
        print("Stage C: no node spectra to assemble"); return
    # one filter interpolator S(lambda) per band
    bands = [(BI.load_filter(str(fd["file"]), conv=FILTER_CONV), fd) for fd in FILTERS]

    # first pass: collect the (logtemp, logg) grid points and a reference mu/wav axis
    info, mu_ref, wav_ref = [], None, None
    for f in files:
        with h5py.File(f, "r") as h:
            lt, lg = float(h["cols/logtemp"][0]), float(h["cols/logg"][0])
            if mu_ref is None:                    # all nodes share the same mu & wav grid
                mu_ref, wav_ref = h["cols/mu"][...], h["cols/wav"][...]
        info.append((lt, lg, f))

    logtemp = np.array(sorted({lt for lt, _, _ in info}))   # unique sorted axes
    logg    = np.array(sorted({lg for _, lg, _ in info}))
    iT = {round(v, 6): k for k, v in enumerate(logtemp)}    # value -> index lookups
    ig = {round(v, 6): k for k, v in enumerate(logg)}
    # one band-flux cube per filter, NaN where a grid point has no model
    flux = {fd["name"]: np.full((len(logtemp), len(logg), len(mu_ref)), np.nan)
            for _, fd in bands}

    # second pass: read each spectrum ONCE, band-integrate through ALL filters
    for lt, lg, f in info:
        with h5py.File(f, "r") as h:
            wav  = h["cols/wav"][...]
            spec = np.exp(h["flux"][0, 0])        # (n_mu, n_wav) Iλ (undo the stored log)
        i, j = iT[round(lt, 6)], ig[round(lg, 6)]
        for band, fd in bands:
            fb  = band(wav)                       # transmission S(λ) on the spectrum's grid
            # AB band-averaged flux density (Bessell & Murphy 2012 A12b), Simpson over λ:
            num = BI.scipy.integrate.simpson(spec * fb * wav, x=wav)        # ∫ Iλ S λ dλ  (per mu)
            den = BI.scipy.integrate.simpson(fb * BI.C_A_PER_S / wav, x=wav)  # ∫ S (c/λ) dλ
            flux[fd["name"]][i, j] = num / den    # <f_nu>(mu) for this (Teff,logg)

    # optional shared mu target (icarus 91): explicit MU list, or copy from MU_REF grid
    mu_target = None
    if MU is not None:
        mu_target = np.asarray(MU, dtype=float)
    elif MU_REF:
        with h5py.File(MU_REF, "r") as h:
            mu_target = h["cols/mu"][...]

    T = np.exp(logtemp)                           # Teff (K) for the print
    for band, fd in bands:                        # write one grid per filter
        fl      = flux[fd["name"]]
        pivot   = float(BI.pivot_wavelength(band, wav_ref))    # band metadata
        missing = int(np.isnan(fl[..., 0]).sum())              # grid points with no model
        logflux, mu_out = np.log(fl), mu_ref       # store log<f_nu>; default native mu
        if mu_target is not None:                  # resample mu (exact: linear in I)
            mu_out = mu_target
            out = np.empty((logflux.shape[0], logflux.shape[1], len(mu_out)))
            for i in range(logflux.shape[0]):
                for j in range(logflux.shape[1]):
                    out[i, j] = np.interp(mu_out, mu_ref, logflux[i, j])  # NaN rows stay NaN
            logflux = out

        outp = fd["out"]
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.exists():
            outp.unlink()                          # overwrite cleanly
        with h5py.File(outp, "w") as fh:           # write the Icarus AtmoGridPhot layout
            d = fh.create_dataset("flux", data=logflux)        # (nT, ng, nmu) = log<f_nu>
            d.attrs["name"] = fd["name"]
            d.attrs["description"] = (fd["desc"] + "-- Specific Intensity Photometry "
                                      "Generated from ATLAS12 on " + Time.now().iso)
            d.attrs["unit"] = "dex(erg / (Angstrom cm2 s))"
            m = fh.create_dataset("meta", dtype="f")           # scalar dataset holding attrs
            m.attrs["zp"], m.attrs["magsys"] = -48.6, "AB"     # AB zeropoint
            m.attrs["ext"], m.attrs["filter"] = fd["ext"], fd["desc"]
            m.attrs["pivot"], m.attrs["Z"] = pivot, 0.0
            grp = fh.create_group("cols", track_order=True)    # the axis datasets
            c = grp.create_dataset("logtemp", data=logtemp)
            c.attrs.update(name="logtemp", description="Log (e base) of temperature", unit="dex(K)")
            c = grp.create_dataset("logg", data=logg)
            c.attrs.update(name="logg", description="Log (10 base) of surface gravity", unit="dex(m / s2)")
            c = grp.create_dataset("mu", data=mu_out)
            c.attrs.update(name="mu", description="cosine of angle of incidence")
        try:                                       # durable flush: survive an abrupt job exit
            fdsc = os.open(str(outp), os.O_RDONLY); os.fsync(fdsc); os.close(fdsc)
        except OSError:
            pass
        print(f"Stage C: wrote {outp}  flux {logflux.shape}  ({missing} NaN)  "
              f"mu={len(mu_out)}  pivot {pivot:.1f}A  ext {fd['ext']}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    argv = sys.argv[1:]
    cfgs = [a for a in argv if not a.startswith("-")]   # positional arg = the config file
    if not cfgs:
        sys.exit("usage: [mpirun -n N] python run_grid.py <config.py> [--assemble-only]")
    load_config(cfgs[0])                          # populate globals from the config

    # Stage C only: serial, durable assemble (run after the mpirun A+B step).
    if "--assemble-only" in argv:
        assemble()
        return

    # MPI context (falls back to a single serial rank if mpi4py is absent)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:
        comm, rank, size = None, 0, 1

    teff_list = frange(*TEFF)                      # expand the [start,stop,step] axes
    logg_list = frange(*LOGG)
    combos = list(product(teff_list, logg_list, FEH))   # every grid point (Teff,logg,feh)

    if rank == 0:                                  # rank 0 prints the run header & makes dirs
        bandnames = ",".join(fd["name"] for fd in FILTERS)
        print(f"GRID  {len(teff_list)} Teff × {len(logg_list)} logg × {len(FEH)} feh "
              f"= {len(combos)} models  |  {size} ranks  |  bands [{bandnames}]")
        print(f"  Teff {teff_list[0]:.0f}..{teff_list[-1]:.0f}  "
              f"logg {logg_list[0]:.1f}..{logg_list[-1]:.1f}")
        print(f"  filters {[fd['file'].name for fd in FILTERS]}  ->  "
              f"SYNTHE {WSTART:.0f}-{WEND:.0f} A (union)  R={RESOLU}")
        MODELS.mkdir(parents=True, exist_ok=True)
        H5DIR.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()                             # all ranks wait until dirs exist

    # ── Stage A: build models (Kurucz-grid start) ───────────────────────────
    for teff, logg, feh in combos[rank::size]:     # round-robin split across ranks
        try:
            status, _ = build_model(teff, logg, feh)
            print(f"[A rank {rank}] {node_name(teff,logg,feh)}: {status}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[A rank {rank}] {node_name(teff,logg,feh)}: ERR {exc}", flush=True)
    if comm is not None:
        comm.Barrier()                             # all models attempted before fallback

    # ── Stage A fallback: retry failures from converged neighbours ───────────
    for rnd in range(FALLBACK_ROUNDS):             # a few passes (neighbours appear as others converge)
        for teff, logg, feh in combos[rank::size]:
            node  = node_name(teff, logg, feh)
            final = MODELS / node / "model_final.dat"
            if final.exists() and final.stat().st_size > 0:   # already built -> skip
                continue
            done = False
            for nb in converged_neighbours(teff, logg, feh, node)[:FALLBACK_NEIGHBOURS]:
                try:
                    status, _ = build_model(teff, logg, feh, start_model=nb)
                except Exception as exc:                           # noqa: BLE001
                    status = f"ERR {exc}"
                if status in ("ok", "warn"):       # converged from this neighbour -> stop trying
                    print(f"[A-fb{rnd} rank {rank}] {node}: {status} (start {nb.parent.name})",
                          flush=True)
                    done = True
                    break
            if not done:
                print(f"[A-fb{rnd} rank {rank}] {node}: still failing "
                      f"(tried {FALLBACK_NEIGHBOURS} neighbours)", flush=True)
        if comm is not None:
            comm.Barrier()

    # ── Stage B: SYNTHE (node-local scratch) ─────────────────────────────────
    models = sorted(MODELS.glob("run_t*/model_final.dat"))   # every converged model
    for mp in models[rank::size]:                  # round-robin split across ranks
        mp   = Path(mp).resolve()
        node = mp.parent.name
        out  = H5DIR / f"{node}.spectro.h5"
        if out.exists() and out.stat().st_size > 0:   # resume-safe: spectrum already built
            continue
        try:
            build_spectrum(mp, out)
            print(f"[B rank {rank}] {node}: ok", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[B rank {rank}] {node}: FAIL {exc}", flush=True)
    if comm is not None:
        comm.Barrier()

    # ── Stage C ──────────────────────────────────────────────────────────────
    # Only assemble here for a serial run (size == 1) — that writes durably.
    # Under mpirun the launcher runs `python run_grid.py <config> --assemble-only`
    # afterwards (avoids the 0-byte mpirun-exit bug).
    if rank == 0:
        if size == 1:
            assemble()
        else:
            print(f"\nStages A+B done. Run Stage C separately (durable):\n"
                  f"  python run_grid.py {cfgs[0]} --assemble-only", flush=True)


if __name__ == "__main__":
    main()
