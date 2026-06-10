#!/usr/bin/env python3
"""Build one ATLAS9 specific-intensity (spectro) grid node end-to-end:

    ATLAS12 model  ->  SYNTHE surface-intensity I(mu)  ->  AtmoGridSpectro HDF5

This is a standalone orchestrator. It does NOT edit the existing pipeline:
  * ATLAS12 model  — invokes run_atlas12.py as a subprocess (full CLI, so --feh
                     and --abund abundance control are available).
  * SYNTHE I(mu)   — imports synthe.py's step functions and runs them with the
                     SURFACE INTENSI card, keeping the mu-resolved spectrum.bin
                     (rotate/broaden/converfer are skipped — they collapse mu).
  * Convert        — Atlas_Reader.ReadSYNTHE + direct h5py write, replicating
                     Atlas_Reader.WriteHDF5Icarus. No compiled Icarus needed
                     (it is built for py3.10; this env is py3.11).

The output HDF5 is the input that Make_Photometry_Files_ATLAS9.py band-integrates
through a filter (e.g. ATLAS c / o) to produce the final ATLAS9.AB.*.h5 grids.

Example
-------
    python build_spectro_grid.py \\
        --teff 5500 --logg 4.0 --feh 0.0 \\
        --wstart 5400 --wend 8400 --resolu 100000 \\
        --workdir spectro_test \\
        --out spectro_test/ATLAS9_t5500g40p00.spectro.h5
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import astropy.constants as const
from astropy.time import Time

# ── local driver modules (in this src/ directory) ──────────────────────────
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

import synthe as S          # SYNTHE step functions (src/synthe.py)

RUN_ATLAS12 = _SRC / "run_atlas12.py"


# ── spectrv fort.7 reader matching THIS build's spectrv.for ─────────────────
# spectrv.for writes per wavelength: WRITE(7)(Q(I),I=1,NMU2) with NMU2=NMU*2,
#   Q(MU)     = RESID*SURF(MU) = SURFI(MU)  -> emergent surface intensity (lines)
#   Q(MU+NMU) = SURF(MU)                    -> continuum intensity
# (icarus-local/Atlas_Reader.py assumes NMU values/record, so it can't read this.)

def read_synthe_intensity(fn):
    """Decode the spectrv surface-intensity binary produced by packages/ATLAS
    spectrv.exe. Returns a dict with SPECINT = first NMU of each record = the
    emergent surface specific intensity I(mu) (Inu, erg/cm^2/s/Hz/sr)."""
    import struct  # to unpack the Fortran unformatted-sequential binary

    with open(fn, "rb") as f:
        # --- HEADER record. Fortran wraps each record with a 4-byte length marker on
        #     BOTH ends; `recl` is the opening marker (= payload byte count). ---
        recl = struct.unpack("i", f.read(4))[0]            # opening record-length marker
        TEFF, GLOG = struct.unpack("2d", f.read(16))       # 2 real*8: Teff, log g
        TITLE = struct.unpack("74s", f.read(74))[0].decode(errors="replace")  # char*74 label
        WBEGIN, DELTAW = struct.unpack("2d", f.read(16))   # real*8: first wavelength (nm), R
        NUMNU, IFSURF, NMU = struct.unpack("3i", f.read(12))  # int*4: #wavelengths, surf flag, #angles
        ANGLE = np.array(struct.unpack("20d", f.read(160)))   # 20 real*8 mu values (only NMU used)
        NEDGE = struct.unpack("i", f.read(4))[0]              # int*4: #continuum edges (unused here)
        WLEDGE = np.array(struct.unpack("377d", f.read(377 * 8)))  # 377 real*8 edges (unused)
        endl = struct.unpack("i", f.read(4))[0]            # closing marker — must equal recl
        if endl != recl:
            raise ValueError(f"header marker mismatch: open={recl} close={endl}")

        # --- DATA records: one per wavelength, each holding NMU*2 doubles =
        #     [SURFI(mu) (line intensity) | continuum(mu)]. We keep the first NMU. ---
        nmu2 = NMU * 2
        payload = nmu2 * 8                                 # bytes of doubles per record
        specint = np.empty((NUMNU, NMU), dtype=np.float64)  # -> (wavelength, mu)
        for i in range(NUMNU):
            m1 = struct.unpack("i", f.read(4))[0]          # opening marker of this record
            q  = struct.unpack(f"{nmu2}d", f.read(payload))  # the 2*NMU doubles
            m2 = struct.unpack("i", f.read(4))[0]          # closing marker
            if m1 != payload or m2 != payload:             # markers verify we stayed in sync
                raise ValueError(
                    f"data record {i}: marker {m1}/{m2} != expected {payload}")
            specint[i, :] = q[:NMU]      # SURFI(MU): emergent line intensity (drop continuum half)

    # rebuild the geometric (constant-R) wavelength axis from WBEGIN(nm) and R; x10 -> Angstrom
    WAV = WBEGIN * (1.0 + 1.0 / DELTAW) ** np.arange(NUMNU) * 10.0   # nm->Angstrom
    return dict(TEFF=TEFF, GLOG=GLOG, TITLE=TITLE, WBEGIN=WBEGIN, DELTAW=DELTAW,
                NUMNU=NUMNU, IFSURF=IFSURF, NMU=NMU, ANGLE=ANGLE, NEDGE=NEDGE,
                WLEDGE=WLEDGE, SPECINT=specint, WAV=WAV)


# ── stage 1: ATLAS12 model (via run_atlas12.py subprocess) ──────────────────

def _node_dirname(teff, logg, feh):
    """Match run_atlas12.py's auto naming: run_t{T}g{logg*10}{p/m}{|feh|*10}."""
    sign = "p" if feh >= 0 else "m"
    return f"run_t{int(teff)}g{int(logg * 10):02d}{sign}{int(abs(feh) * 10):02d}"


def build_model(teff, logg, feh, workdir, vturb=2.0, iters=45,
                molecules=False, abund=None):
    """Run run_atlas12.py for one node. Returns path to model_final.dat
    (skips the run if the model already exists)."""
    workdir = Path(workdir).resolve()
    rundir  = workdir / _node_dirname(teff, logg, feh)
    final   = rundir / "model_final.dat"

    if final.exists() and final.stat().st_size > 0:
        print(f"[ATLAS12] model already exists: {final}")
        return final

    cmd = [sys.executable, str(RUN_ATLAS12),
           "--teff", repr(teff), "--logg", repr(logg), "--feh", repr(feh),
           "--vturb", repr(vturb), "--iter", str(iters),
           "--outdir", str(workdir)]
    if molecules:
        cmd.append("--molecules")
    if abund:
        cmd += ["--abund", str(abund)]

    print(f"[ATLAS12] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not (final.exists() and final.stat().st_size > 0):
        sys.exit(f"ERROR: ATLAS12 did not produce {final}")
    return final


# ── stage 2: SYNTHE surface-intensity (imported synthe.py steps) ────────────

def build_intensity_bin(rundir, model_path, wstart, wend, resolu,
                        vturb, molecules, linelists, airvac, mu_angles=None):
    """xnfpelsyn -> synbeg -> rgfall -> [molec] -> synthe -> spectrv with the
    SURFACE INTENSI card; keep the mu-resolved binary spectrum.bin.
    mu_angles: optional explicit cos-angle list (<=~12 for a card-safe run);
    default None uses synthe.py's native 17."""
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    spectrum_bin = rundir / "spectrum.bin"
    if spectrum_bin.exists() and spectrum_bin.stat().st_size > 0:
        print(f"[SYNTHE] spectrum.bin exists, skipping SYNTHE: {spectrum_bin}")
        return spectrum_bin

    synthe_model = rundir / "model_synthe.mod"
    S.make_synthe_model(model_path, synthe_model,
                        molecules=molecules, intensi=True, mu_angles=mu_angles)

    print("\n[SYNTHE 1/6] xnfpelsyn")
    opacity_model = S.step_xnfpelsyn(rundir, synthe_model)
    print("\n[SYNTHE 2/6] synbeg")
    S.step_synbeg(rundir, wstart, wend, resolu, vturb, airvac)
    print(f"\n[SYNTHE 3/6] rgfalllinesnew — {len(linelists)} atomic list(s)")
    for ll in linelists:
        S.step_rgfalllinesnew(rundir, ll)
    if molecules:
        print("\n[SYNTHE 4/6] molecular lines")
        for mol_file in sorted(S.MOL_DB.glob("*.asc")):
            S.step_rmolec(rundir, mol_file)
        S.step_rschwenk(rundir)
        S.step_rh2ofast(rundir)
    else:
        print("\n[SYNTHE 4/6] molecular lines — skipped")
    print("\n[SYNTHE 5/6] synthe")
    S.step_synthe(rundir, opacity_model)
    print("\n[SYNTHE 6/6] spectrv — surface intensity I(mu)")
    spectrum_bin = S.step_spectrv(rundir, synthe_model)   # -> spectrum.bin (fort.7)

    S._cleanup_fort(rundir)
    return spectrum_bin


# ── stage 3: convert spectrv binary -> AtmoGridSpectro HDF5 ─────────────────

def write_spectro_h5(bin_path, out_path, overwrite=True):
    """Decode the spectrv binary and write an AtmoGridSpectro-format HDF5,
    replicating Atlas_Reader.WriteHDF5Icarus (in_nu=False).
    """
  # 1. Decode the binary (read_synthe_intensity)

  # The spectrum.bin is a Fortran "unformatted sequential" file: every record is wrapped in a 4-byte length marker on each side. First record = header:
  # TEFF, GLOG, TITLE, WBEGIN, DELTAW, NUMNU, IFSURF, NMU, ANGLE[20], NEDGE, WLEDGE[377]
  # Then NUMNU (=44183) records follow, one per wavelength, each holding 34 doubles = [17 intensities | 17 continua]. The reader reads each record and keeps the first 17 (the actual line intensity SURFI):
  # specint[i, :] = q[:NMU]      # drop the continuum half

  # 2. The physics: Iν → Iλ → log

  # SYNTHE stores intensity per unit frequency (I_ν, erg/cm²/s/Hz/sr). Filters and Icarus work per unit wavelength (I_λ, per Å). The conversion uses I_λ = I_ν · c/λ²:
  # grid = np.log(SPECINT.T * c·1e10 / wav**2)
  # - c·1e10 is the speed of light in Å/s, so the result is per Å.
  # - We store log(I_λ) — that's why the flux unit is dex(erg/(Å cm² s sr)) and the values are ~16.9 instead of ~2×10⁷. Log keeps the huge dynamic range (cool limb vs hot core, line cores vs continuum) numerically tame, and interpolation is smoother in log-space.

  # You can see the physics in the file: at 6000 Å, log I = 16.92 at μ=1 (disc centre) vs 15.96 at μ=0.01 (limb) → limb darkening, the centre is e^(16.92−15.96) ≈ 2.6× brighter.

  # 3. The μ axis gets flipped

  # SYNTHE emits angles descending: [1.0, 0.9, ... 0.01]. The code detects that and reverses both the axis and the data together so μ runs ascending (0.01 → 1.0, which you see in the output):
  # elif np.all(np.diff(mu) < 0):
  #     mu, grid = mu[::-1], grid[::-1]
  # Icarus expects monotonically increasing axes, so this ordering matters. μ = cos(angle from disc centre): μ=1 is the centre, μ→0 is the limb.

  # 4. The wavelength axis is reconstructed, not stored per-point

  # SYNTHE samples on a geometric grid (constant resolving power R), so wavelengths aren't written out — they're regenerated from two header numbers:
  # WAV = WBEGIN · (1 + 1/R)^[0,1,2,…,NUMNU−1] · 10     # nm → Å
  # WBEGIN=540.0 nm, R=100000 → 44183 points from 5400→8400 Å, each step ×(1.00001).

  # The final layout

  # It's written as a 4-D cube flux[logtemp, logg, mu, wav] = (1, 1, 17, 44183) for a single node, plus the axis datasets in cols/ and header info (R, wav0) in meta. The leading (1,1) is the placeholder for this node's single temperature/gravity — Stage C stacks 287 of these into the full (41, 7, …) grid.

  # So in one sentence: read the binary → convert per-Hz to per-Å and take the log → put μ in increasing order → rebuild the wavelength grid → save as a labelled 4-D HDF5 cube. That cube is the "full spectrum at every viewing angle" that the band-integration step then collapses through the filter.
    import h5py

    data = read_synthe_intensity(str(bin_path))
    nmu  = data["NMU"]     # Number of wavelength points (the length of the spectrum).
    wav  = np.atleast_1d(data["WAV"])                       # (n_wav,)  Angstrom

    # Inu -> Ilambda, then log. SPECINT is (n_wav, n_mu); -> (n_mu, n_wav).
    grid = np.log(data["SPECINT"].T * const.c.value * 1e10 / wav**2)
    unit = "dex(erg / (Angstrom cm2 s sr))"

    mu = np.atleast_1d(data["ANGLE"][:nmu])
    if np.all(np.diff(mu) > 0):
        pass
    elif np.all(np.diff(mu) < 0):
        mu, grid = mu[::-1], grid[::-1]
    else:
        order = mu.argsort()
        mu, grid = mu[order], grid[order]

    logtemp = np.atleast_1d(np.log(data["TEFF"]))
    logg    = np.atleast_1d(data["GLOG"])
    data4d  = grid[None, None, ...]                         # (1, 1, n_mu, n_wav)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        out_path.unlink()

    with h5py.File(out_path, "w") as f:
        dflux = f.create_dataset("flux", data=data4d)
        dflux.attrs["name"] = "ATLAS12"
        dflux.attrs["description"] = (
            data["TITLE"].strip()
            + " -- Specific Intensity Generated on {}".format(Time.now().iso))
        dflux.attrs["unit"] = unit

        dmeta = f.create_dataset("meta", dtype="f")
        dmeta.attrs["wav0"] = data["WBEGIN"]
        dmeta.attrs["R"]    = data["DELTAW"]

        grp = f.create_group("cols", track_order=True)
        c = grp.create_dataset("logtemp", data=logtemp)
        c.attrs.update(name="logtemp",
                       description="Log (e base) of temperature", unit="dex(K)")
        c = grp.create_dataset("logg", data=logg)
        c.attrs.update(name="logg",
                       description="Log (10 base) of surface gravity",
                       unit="dex(m / s2)")
        c = grp.create_dataset("mu", data=mu)
        c.attrs.update(name="mu", description="cosine of angle of incidence")
        c = grp.create_dataset("wav", data=wav)
        c.attrs.update(name="wav", description="Wavelength", unit="Angstrom")

    # Durably flush to storage so the spectrum survives an abrupt job exit:
    # lustre buffers writes, so an mpirun rank's last-written file can land
    # 0-byte / corrupt if the SLURM job ends before the buffer syncs to the OSTs.
    import os
    try:
        fd = os.open(str(out_path), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass

    print(f"\n[CONVERT] wrote {out_path}")
    print(f"  flux (logtemp, logg, mu, wav) = {data4d.shape}")
    print(f"  T={data['TEFF']:.1f} K  GLOG={data['GLOG']:.3f}  "
          f"mu={len(mu)}  wav={len(wav)} ({wav.min():.1f}..{wav.max():.1f} A)")
    return out_path


def write_spectro_h5_merged(bin_paths, out_path, overwrite=True):
    """Merge several spectrv binaries (same model & wavelength grid, different mu
    subsets) into ONE AtmoGridSpectro h5 covering all their mu angles. Used for
    the 'native' mu mode, where the target angles are computed in card-safe
    batches and stitched together (no interpolation)."""
    import os, h5py
    datas = [read_synthe_intensity(str(b)) for b in bin_paths]
    d0  = datas[0]
    wav = np.atleast_1d(d0["WAV"])
    mus, grids = [], []
    for d in datas:
        nmu = d["NMU"]
        mus.append(np.atleast_1d(d["ANGLE"][:nmu]))
        grids.append(np.log(d["SPECINT"].T * const.c.value * 1e10 / wav**2))  # (nmu,nwav)
    mu   = np.concatenate(mus)
    grid = np.concatenate(grids, axis=0)
    order = np.argsort(mu)
    mu, grid = mu[order], grid[order]
    data4d = grid[None, None, ...]
    logtemp = np.atleast_1d(np.log(d0["TEFF"]))
    logg    = np.atleast_1d(d0["GLOG"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        out_path.unlink()
    with h5py.File(out_path, "w") as f:
        dflux = f.create_dataset("flux", data=data4d)
        dflux.attrs["name"] = "ATLAS12"
        dflux.attrs["description"] = (d0["TITLE"].strip()
            + " -- Specific Intensity Generated on {}".format(Time.now().iso))
        dflux.attrs["unit"] = "dex(erg / (Angstrom cm2 s sr))"
        dmeta = f.create_dataset("meta", dtype="f")
        dmeta.attrs["wav0"] = d0["WBEGIN"]; dmeta.attrs["R"] = d0["DELTAW"]
        grp = f.create_group("cols", track_order=True)
        c = grp.create_dataset("logtemp", data=logtemp)
        c.attrs.update(name="logtemp", description="Log (e base) of temperature", unit="dex(K)")
        c = grp.create_dataset("logg", data=logg)
        c.attrs.update(name="logg", description="Log (10 base) of surface gravity", unit="dex(m / s2)")
        c = grp.create_dataset("mu", data=mu)
        c.attrs.update(name="mu", description="cosine of angle of incidence")
        c = grp.create_dataset("wav", data=wav)
        c.attrs.update(name="wav", description="Wavelength", unit="Angstrom")
    try:
        fd = os.open(str(out_path), os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError:
        pass
    print(f"[CONVERT-merged] {out_path.name}: {len(mu)} native mu, {len(wav)} wav")
    return out_path


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ATLAS12 (model) parameters
    g1 = p.add_argument_group("ATLAS12 model")
    g1.add_argument("--teff", type=float, required=True, help="Effective temperature (K)")
    g1.add_argument("--logg", type=float, required=True, help="log g (cgs)")
    g1.add_argument("--feh",  type=float, default=0.0, help="[M/H] (default 0.0)")
    g1.add_argument("--vturb", type=float, default=2.0, help="Model microturbulence km/s (default 2.0)")
    g1.add_argument("--iter", type=int, default=45, help="ATLAS12 iterations (default 45)")
    g1.add_argument("--model-molecules", action="store_true",
                    help="ATLAS12 molecular equilibrium")
    g1.add_argument("--abund", default=None, help="Abundance file (Z log_abundance per line)")
    g1.add_argument("--model", default=None,
                    help="Use this existing model_final.dat instead of running ATLAS12")
    # SYNTHE parameters
    g2 = p.add_argument_group("SYNTHE")
    g2.add_argument("--wstart", type=float, required=True, help="Start wavelength (A)")
    g2.add_argument("--wend",   type=float, required=True, help="End wavelength (A)")
    g2.add_argument("--resolu", type=float, default=100000.0, help="Resolving power (default 1e5)")
    g2.add_argument("--synthe-vturb", type=float, default=0.0,
                    help="Extra SYNTHE microturbulence km/s (default 0)")
    g2.add_argument("--synthe-molecules", action="store_true",
                    help="Include molecular line lists in SYNTHE")
    g2.add_argument("--linelist", action="append", default=None,
                    help="Atomic line list(s); default gfall.dat")
    g2.add_argument("--vac", action="store_true", help="Vacuum wavelengths (default air)")
    # output
    g3 = p.add_argument_group("output")
    g3.add_argument("--workdir", default="spectro_build",
                    help="Base dir for ATLAS12 node + SYNTHE run (default spectro_build)")
    g3.add_argument("--out", "-o", required=True, help="Output spectro grid HDF5 path")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    airvac  = "VAC" if args.vac else "AIR"

    # ── stage 1: model ───────────────────────────────────────────────────────
    print("=" * 70)
    print(f"NODE  Teff={args.teff:.0f}  logg={args.logg:.2f}  [M/H]={args.feh:+.1f}")
    print("=" * 70)
    if args.model:
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            p.error(f"--model not found: {model_path}")
        print(f"[ATLAS12] using supplied model: {model_path}")
    else:
        model_path = build_model(args.teff, args.logg, args.feh, workdir,
                                 vturb=args.vturb, iters=args.iter,
                                 molecules=args.model_molecules, abund=args.abund)

    # ── stage 2: SYNTHE surface intensity ────────────────────────────────────
    linelists = ([Path(f).resolve() for f in args.linelist]
                 if args.linelist else [S.DEFAULT_LINELIST])
    for ll in linelists:
        if not ll.exists():
            p.error(f"Line list not found: {ll}")

    synthe_rundir = (model_path.parent
                     / f"synthe_intensi_{args.wstart:.0f}-{args.wend:.0f}")
    spectrum_bin = build_intensity_bin(
        synthe_rundir, model_path, args.wstart, args.wend, args.resolu,
        args.synthe_vturb, args.synthe_molecules, linelists, airvac)

    # ── stage 3: convert ─────────────────────────────────────────────────────
    write_spectro_h5(spectrum_bin, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
