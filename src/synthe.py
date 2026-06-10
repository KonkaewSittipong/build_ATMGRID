#!/usr/bin/env python3
"""Run the Kurucz/Castelli SYNTHE spectral synthesis pipeline.

Pipeline:
    xnfpelsyn → synbeg → rgfalllinesnew → [rmolecasc / rschwenk / rh2ofast]
              → synthe → spectrv → [rotate] → [broaden] → converfsynnmtoa

Output is written inside the ATLAS12 run directory that contains the model:
    run_t5500g46m10/synthe_5000-5100/spectrum.dat

Single model / single broadening
---------------------------------
    python synthe.py --model run_t5500g46m10/model_final.dat --wstart 5000 --wend 5100
    python synthe.py --model run_t5500g46m10/model_final.dat \\
        --wstart 4000 --wend 7000 --resolu 500000 --vsini 50 --broaden-R 48000
    python synthe.py --model run_t5500g46m10/model_final.dat \\
        --wstart 7000 --wend 7200 --molecules --linelist /path/to/extra_lines.dat

Grid (multiple models and/or broadening values)
------------------------------------------------
    # One model, four vsini values → 4 spectra inside run_t5500g46m10/
    python synthe.py --model run_t5500g46m10/model_final.dat \\
        --vsini 0 50 100 150 --wstart 5000 --wend 5200

    # Glob all models, two vsini values → N × 2 spectra, each in its own model dir
    python synthe.py --model run_t*/model_final.dat \\
        --vsini 0 50 --wstart 5000 --wend 5200

    # Models × vsini × instrumental R
    python synthe.py --model run_t*/model_final.dat \\
        --vsini 0 50 100 --broaden-R 48000 --wstart 5000 --wend 5200

    # MPI: distribute combos over 8 cores
    mpirun -n 8 python synthe.py --model run_t*/model_final.dat \\
        --vsini 0 50 100 150 --wstart 5000 --wend 5200
"""

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
# This driver lives in build_ATMGRID/src/, but the SYNTHE executables and line
# lists live in the packages/ATLAS install — point there.
HERE       = Path("/lustre/MSSP/sittipong/packages/ATLAS")
SYNTHE_DIR = HERE / "synthe"
LINES_DIR  = HERE / "lines"
LINE_DB    = LINES_DIR / "atomic"
MOL_DB     = LINES_DIR / "molecules"

XNFPELSYN  = SYNTHE_DIR / "xnfpelsyn.exe"
SYNBEG     = SYNTHE_DIR / "synbeg.exe"
RGFALL     = SYNTHE_DIR / "rgfalllinesnew.exe"
RMOLEC     = SYNTHE_DIR / "rmolecasc.exe"
RSCHWENK   = SYNTHE_DIR / "rschwenk.exe"
RH2OFAST   = SYNTHE_DIR / "rh2ofast.exe"
SYNTHE_EXE = SYNTHE_DIR / "synthe.exe"
SPECTRV    = SYNTHE_DIR / "spectrv.exe"
ROTATE     = SYNTHE_DIR / "rotate.exe"
BROADEN    = SYNTHE_DIR / "broaden.exe"
CONVERFER  = SYNTHE_DIR / "converfsynnmtoa.exe"

DEFAULT_LINELIST = LINE_DB / "gfall.dat"


# ── helpers ───────────────────────────────────────────────────────────────────

def symlink(src, dst):
    dst = Path(dst)
    src = Path(src).resolve()
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _cleanup_fort(rundir, keep=()):
    """Remove all fort.N files (symlinks and regular) except those in keep."""
    for f in Path(rundir).glob("fort.*"):
        if f.name in keep:
            continue
        if f.is_symlink() or f.is_file():
            f.unlink()


def run_exe(exe, stdin_text=None, stdin_path=None, logfile=None, cwd=None):
    """Run an executable. Returns (returncode, stdout_text).
    stdin_text: string to pipe into stdin.
    stdin_path: file to redirect as stdin (used instead of stdin_text).
    """
    cwd = Path(cwd) if cwd else Path.cwd()
    if stdin_path is not None:
        fh = open(stdin_path, "rb")
        proc = subprocess.run(
            [str(exe)], stdin=fh,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
        )
        fh.close()
    elif stdin_text is not None:
        proc = subprocess.run(
            [str(exe)], input=stdin_text.encode(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
        )
    else:
        proc = subprocess.run(
            [str(exe)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
        )

    out = proc.stdout.decode(errors="replace")
    if logfile:
        Path(logfile).write_text(out)
    return proc.returncode, out


# ── synbeg input ──────────────────────────────────────────────────────────────

def synbeg_card(wstart_ang, wend_ang, resolu, vturb, airvac="AIR",
                ifnlte=0, linout=10, cutoff=0.001, ifpred=0, nread=0):
    """Build the fixed-format synbeg stdin card.

    synbeg uses nm internally; converfsynnmtoa multiplies by 10 for Å output.
    wstart_ang / wend_ang are the user-facing values in Å; divide by 10 → nm.

    FORMAT(A3,7X,4F10.4,I3,I7,F10.5,2I5)
    """
    wlbeg = wstart_ang / 10.0   # Å → nm
    wlend = wend_ang   / 10.0
    # F10.4 overflows for resolu>=10000 (500000.0000 is 11 chars).
    # Use .1f for resolu so it fits in 10 chars up to 10 million.
    return (
        f"{airvac:<3s}{'':7s}"
        f"{wlbeg:10.4f}{wlend:10.4f}{resolu:10.1f}{vturb:10.4f}"
        f"{ifnlte:3d}{linout:7d}{cutoff:10.5f}{ifpred:5d}{nread:5d}\n"
    )


# ── pipeline steps ─────────────────────────────────────────────────────────────

def step_xnfpelsyn(rundir, model_path):
    """Run xnfpelsyn to produce the opacity-sampled model (fort.10 → opacity_model.dat)."""
    # Remove stale fort.10 / opacity_model.dat from any previous failed run to
    # avoid circular symlinks (fort.10 symlink → opacity_model.dat → rename loop).
    for stale in (rundir / "fort.10", rundir / "opacity_model.dat"):
        if stale.is_symlink() or stale.exists():
            stale.unlink()

    symlink(LINES_DIR / "molecules.dat", rundir / "fort.2")
    symlink(LINES_DIR / "continua.dat",  rundir / "fort.17")
    rc, out = run_exe(XNFPELSYN, stdin_path=model_path,
                      logfile=rundir / "xnfpelsyn.log", cwd=rundir)
    _cleanup_fort(rundir, keep=("fort.10",))

    opacity_model = rundir / "opacity_model.dat"
    fort10 = rundir / "fort.10"
    if fort10.is_file() and not fort10.is_symlink() and fort10.stat().st_size > 0:
        fort10.rename(opacity_model)
    else:
        sys.exit("ERROR: xnfpelsyn did not produce fort.10 — check xnfpelsyn.log")
    print(f"  xnfpelsyn: {opacity_model.name} ({opacity_model.stat().st_size:,} bytes)")
    return opacity_model


def step_synbeg(rundir, wstart, wend, resolu, vturb, airvac):
    """Run synbeg to initialise binary line tapes (fort.12/14/19/20) and fort.93."""
    card = synbeg_card(wstart, wend, resolu, vturb, airvac=airvac)
    rc, out = run_exe(SYNBEG, stdin_text=card,
                      logfile=rundir / "synbeg.log", cwd=rundir)
    if rc != 0:
        sys.exit(f"ERROR: synbeg exited with code {rc} — check synbeg.log")
    for tape in ("fort.12", "fort.14", "fort.19", "fort.20", "fort.93"):
        p = rundir / tape
        if not p.exists():
            sys.exit(f"ERROR: synbeg did not create {tape} — check synbeg.log")
    print(f"  synbeg: {airvac} {wstart:.1f}–{wend:.1f} Å  R={resolu:.0f}  vturb={vturb} km/s")


def step_rgfalllinesnew(rundir, linelist_path):
    """Run rgfalllinesnew once for a given atomic line list file (fort.11)."""
    symlink(linelist_path, rundir / "fort.11")
    rc, out = run_exe(RGFALL,
                      logfile=rundir / f"rgfall_{Path(linelist_path).stem}.log",
                      cwd=rundir)
    (rundir / "fort.11").unlink(missing_ok=True)
    if rc != 0:
        sys.exit(f"ERROR: rgfalllinesnew failed for {linelist_path} — check log")
    print(f"  rgfalllinesnew: {Path(linelist_path).name}")


def step_rmolec(rundir, mol_path):
    """Run rmolecasc once for an ASCII molecular line list file (fort.11)."""
    symlink(mol_path, rundir / "fort.11")
    rc, out = run_exe(RMOLEC,
                      logfile=rundir / f"rmolec_{Path(mol_path).stem}.log",
                      cwd=rundir)
    (rundir / "fort.11").unlink(missing_ok=True)
    if rc != 0:
        print(f"  WARNING: rmolecasc returned {rc} for {mol_path.name}")
    else:
        print(f"  rmolecasc: {mol_path.name}")


def step_rschwenk(rundir):
    """Run rschwenk for the TiO Schwenke line list.

    fort.11 = tioschwenke.bin  (packed line list)
    fort.48 = tioschwenke_energy.bin  (energy levels built by eschwbin.for)
    """
    tio = MOL_DB / "tioschwenke.bin"
    if not tio.exists():
        tio = LINES_DIR / "tioschwenke.bin"
    if not tio.exists():
        print("  rschwenk: skipped (tioschwenke.bin not found)")
        return
    energy = LINES_DIR / "tioschwenke_energy.bin"
    if not energy.exists():
        print("  rschwenk: skipped (tioschwenke_energy.bin not found — build with eschwbin.for)")
        return
    symlink(tio,    rundir / "fort.11")
    symlink(energy, rundir / "fort.48")
    rc, out = run_exe(RSCHWENK,
                      logfile=rundir / "rschwenk.log", cwd=rundir)
    (rundir / "fort.11").unlink(missing_ok=True)
    (rundir / "fort.48").unlink(missing_ok=True)
    if rc != 0:
        print(f"  WARNING: rschwenk returned {rc}")
    else:
        print("  rschwenk: TiO Schwenke done")


def step_rh2ofast(rundir):
    """Run rh2ofast for the H2O line list."""
    h2o = LINES_DIR / "h2ofastfix.bin"
    if not h2o.exists():
        print("  rh2ofast: skipped (h2ofastfix.bin not found)")
        return
    symlink(h2o, rundir / "fort.11")
    rc, out = run_exe(RH2OFAST,
                      logfile=rundir / "rh2ofast.log", cwd=rundir)
    (rundir / "fort.11").unlink(missing_ok=True)
    if rc != 0:
        print(f"  WARNING: rh2ofast returned {rc}")
    else:
        print("  rh2ofast: H2O done")


HE1_STARK = LINES_DIR / "he1stark_dummy.dat"

def step_synthe(rundir, opacity_model):
    """Run synthe to compute opacity spectra."""
    symlink(opacity_model, rundir / "fort.10")
    symlink(LINES_DIR / "molecules.dat", rundir / "fort.2")
    # fort.18 = He I Stark broadening table (4471, 4026, 4387, 4921 Å).
    # Required whenever any of those lines falls in the wavelength range.
    # Dummy file gives zero Stark broadening — valid for Teff<~15000 K where
    # He I lines are weak; for hot stars real Stark data should be supplied.
    if HE1_STARK.exists():
        symlink(HE1_STARK, rundir / "fort.18")
    rc, out = run_exe(SYNTHE_EXE,
                      logfile=rundir / "synthe.log", cwd=rundir)
    (rundir / "fort.2").unlink(missing_ok=True)
    # fort.10 stays — spectrv also opens it at OPEN(UNIT=10,...)
    if rc != 0:
        sys.exit(f"ERROR: synthe exited with code {rc} — check synthe.log")
    print("  synthe: opacity spectra computed")


_MU_ANGLES = "1.,.9,.8,.7,.6,.5,.4,.3,.25,.2,.15,.125,.1,.075,.05,.025,.01"


def _fmt_angle(a):
    """Compact Fortran-style angle string (drop the leading zero) so SURFACE
    INTENSI cards stay short: 0.9->'.9', 0.0175->'.0175', 1.0->'1.'. With leading
    zeros the card overflows the ~80-col limit and SYNTHE truncates the smallest
    angles (max ~12-13 angles per card)."""
    s = f"{a:g}"
    if s.startswith("0."):
        s = s[1:]
    elif "." not in s:
        s = s + "."
    return s


def make_synthe_model(model_path, out_path, molecules=False, intensi=False,
                      mu_angles=None):
    """Prepend the control cards Castelli requires to the ATLAS12 model.

    The resulting .mod file is used for both xnfpelsyn and spectrv.
    Without these cards READIN leaves IFSURF=0 and spectrv produces zeros.

    intensi=False: SURFACE FLUX (NMU=1).
    intensi=True:  SURFACE INTENSI with mu_angles (default: the native 17).
        mu_angles: optional list of cos(angle) values (<=20, and few enough that
        the compact card stays under ~80 cols, i.e. ~12-13 max).
    """
    mol_cards = "READ MOLECULES\nMOLECULES ON\n" if molecules else ""
    if intensi:
        if mu_angles is None:
            surf_card = f"SURFACE INTENSI 17 {_MU_ANGLES}\n"
        else:
            astr = ",".join(_fmt_angle(a) for a in mu_angles)
            surf_card = f"SURFACE INTENSI {len(mu_angles)} {astr}\n"
    else:
        surf_card = "SURFACE FLUX\n"
    header = (
        surf_card
        + "ITERATIONS 1 PRINT 2 PUNCH 2\n"
          "CORRECTION OFF\n"
          "PRESSURE OFF\n"
        + mol_cards
    )
    Path(out_path).write_text(header + Path(model_path).read_text())


def step_spectrv(rundir, synthe_model):
    """Run spectrv. synthe_model must already have the control cards prepended."""
    fort25 = rundir / "fort.25"
    fort25.write_text(
        "       0.0       0.       1.       0.       0.       0.       0.       0.\n"
        "       0.\n"
    )
    symlink(LINES_DIR / "molecules.dat", rundir / "fort.2")
    rc, out = run_exe(SPECTRV, stdin_path=synthe_model,
                      logfile=rundir / "spectrv.log", cwd=rundir)
    (rundir / "fort.2").unlink(missing_ok=True)
    if rc != 0:
        sys.exit(f"ERROR: spectrv exited with code {rc} — check spectrv.log")
    fort7 = rundir / "fort.7"
    if not fort7.exists() or fort7.stat().st_size == 0:
        sys.exit("ERROR: spectrv did not produce fort.7 — check spectrv.log")
    spectrum = rundir / "spectrum.bin"
    fort7.rename(spectrum)
    print(f"  spectrv: {spectrum.name} ({spectrum.stat().st_size:,} bytes)")
    return spectrum


def step_rotate(rundir, spectrum, vsini):
    """Run rotate for rotational broadening. Returns new spectrum path."""
    symlink(spectrum, rundir / "fort.1")
    stdin = f"    1    0\n{vsini:10.1f}\n"
    rc, out = run_exe(ROTATE, stdin_text=stdin,
                      logfile=rundir / "rotate.log", cwd=rundir)
    (rundir / "fort.1").unlink(missing_ok=True)
    rot = rundir / "ROT1"
    if rc != 0 or not rot.exists():
        sys.exit(f"ERROR: rotate failed — check rotate.log")
    rotated = rundir / "spectrum_rot.bin"
    rot.rename(rotated)
    print(f"  rotate: vsini={vsini} km/s → {rotated.name}")
    return rotated


def step_broaden(rundir, spectrum, resolu, label="broad", logname="broaden.log"):
    """Run broaden for Gaussian broadening in R space. Returns new spectrum path."""
    symlink(spectrum, rundir / "fort.21")
    stdin = f"GAUSSIAN  {resolu:10.1f}RESOLUTION\n"
    rc, out = run_exe(BROADEN, stdin_text=stdin,
                      logfile=rundir / logname, cwd=rundir)
    (rundir / "fort.21").unlink(missing_ok=True)
    fort22 = rundir / "fort.22"
    if rc != 0 or not fort22.exists():
        sys.exit(f"ERROR: broaden failed — check {logname}")
    broadened = rundir / f"spectrum_{label}.bin"
    fort22.rename(broadened)
    print(f"  broaden: R={resolu:.0f} → {broadened.name}")
    return broadened


_C_KMS = 299792.458  # speed of light km/s

def step_vmacro(rundir, spectrum, vmacro_kms):
    """Apply macroturbulence as a Gaussian in velocity space (FWHM = vmacro_kms).
    Approximated as R-space Gaussian with R = c / vmacro."""
    r_macro = _C_KMS / vmacro_kms
    return step_broaden(rundir, spectrum, r_macro,
                        label="vmacro", logname="vmacro.log")


def step_converfer(rundir, spectrum, outfile):
    """Run converfsynnmtoa to convert binary spectrum to ASCII."""
    symlink(spectrum, rundir / "fort.1")
    rc, out = run_exe(CONVERFER,
                      logfile=rundir / "converfer.log", cwd=rundir)
    (rundir / "fort.1").unlink(missing_ok=True)
    fort2 = rundir / "fort.2"
    if rc != 0 or not fort2.exists():
        sys.exit("ERROR: converfsynnmtoa failed — check converfer.log")
    fort2.rename(outfile)
    print(f"  converfer: ASCII spectrum → {outfile.name}")


def _cleanup_intermediates(rundir):
    """Remove intermediate files after a successful run, keeping only spectrum.dat."""
    rundir = Path(rundir)
    patterns = ["*.log", "*.bin", "*.mod", "fort.*"]
    for pat in patterns:
        for f in rundir.glob(pat):
            if f.is_file() or f.is_symlink():
                f.unlink()
    for name in ("opacity_model.dat", "fort.93", "fort.25"):
        p = rundir / name
        if p.exists() or p.is_symlink():
            p.unlink()
    print("  clean: intermediate files removed")


# ── single-point pipeline (shared by single mode and grid mode) ───────────────

def _run_one(rundir, model_path, wstart, wend, resolu, vturb,
             vsini, vmacro, broaden_R, molecules, linelists, airvac, clean=False):
    """Run the complete SYNTHE pipeline for one (model, broadening) combination."""
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    synthe_model = rundir / "model_synthe.mod"
    make_synthe_model(model_path, synthe_model,
                      molecules=molecules, intensi=(vsini > 0))

    print("\n[1/7] xnfpelsyn — building opacity model")
    opacity_model = step_xnfpelsyn(rundir, synthe_model)

    print("\n[2/7] synbeg — initialising line tapes")
    step_synbeg(rundir, wstart, wend, resolu, vturb, airvac)

    print(f"\n[3/7] rgfalllinesnew — reading {len(linelists)} atomic line list(s)")
    for ll in linelists:
        step_rgfalllinesnew(rundir, ll)

    if molecules:
        print("\n[4/7] molecular lines")
        for mol_file in sorted(MOL_DB.glob("*.asc")):
            step_rmolec(rundir, mol_file)
        step_rschwenk(rundir)
        step_rh2ofast(rundir)
    else:
        print("\n[4/7] molecular lines — skipped (use --molecules to enable)")

    print("\n[5/7] synthe — computing opacity spectra")
    step_synthe(rundir, opacity_model)

    print("\n[6/7] spectrv — integrating emergent flux")
    spectrum = step_spectrv(rundir, synthe_model)

    if vsini > 0:
        print(f"\n[6b] rotate — vsini={vsini} km/s")
        spectrum = step_rotate(rundir, spectrum, vsini)
    if vmacro > 0:
        print(f"\n[6c] vmacro — {vmacro} km/s FWHM (R={_C_KMS/vmacro:.0f})")
        spectrum = step_vmacro(rundir, spectrum, vmacro)
    if broaden_R > 0:
        print(f"\n[6d] broaden — R={broaden_R:.0f}")
        spectrum = step_broaden(rundir, spectrum, broaden_R)

    print("\n[7/7] converfsynnmtoa — ASCII spectrum")
    outfile = rundir / "spectrum.dat"
    step_converfer(rundir, spectrum, outfile)
    _cleanup_fort(rundir)
    if clean:
        _cleanup_intermediates(rundir)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SYNTHE spectral synthesis pipeline wrapper"
    )
    p.add_argument("--model",  "-m", nargs="+", default=[],
                   help="ATLAS12 model atmosphere file(s) — pass multiple for a grid")
    p.add_argument("--model-list", metavar="FILE", default=None,
                   help="Text file with one model path per line (avoids ARG_MAX limits)")
    p.add_argument("--wstart", type=float, required=True,
                   help="Start wavelength (Å)")
    p.add_argument("--wend",   type=float, required=True,
                   help="End wavelength (Å)")
    p.add_argument("--resolu", type=float, default=500000.0,
                   help="Resolving power λ/Δλ (default 500000)")
    p.add_argument("--vturb",  type=float, default=0.0,
                   help="Extra microturbulence for line broadening (km/s, default 0)")
    p.add_argument("--vsini",  type=float, nargs="+", default=[0.0],
                   help="vsini value(s) km/s — pass multiple for a grid (0 = skip)")
    p.add_argument("--broaden-R", type=float, nargs="+", default=[0.0],
                   dest="broaden_R",
                   help="Instrumental broadening R value(s) — pass multiple for a grid (0 = skip)")
    p.add_argument("--vmacro", type=float, nargs="+", default=[0.0],
                   help="Macroturbulence FWHM value(s) km/s — pass multiple for a grid (0 = skip)")
    p.add_argument("--molecules", action="store_true",
                   help="Include molecular lines (TiO via rschwenk, H2O via rh2ofast, "
                        "diatomics via rmolecasc)")
    p.add_argument("--linelist", action="append", default=None, metavar="FILE",
                   help="Atomic line list file(s) in Kurucz ASCII format "
                        f"(default: {DEFAULT_LINELIST}). Repeat for multiple files.")
    p.add_argument("--vac", action="store_true",
                   help="Use vacuum wavelengths (default: air)")
    p.add_argument("--clean", action="store_true",
                   help="Remove intermediate files after each run, keeping only spectrum.dat")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: synthe_<wstart>-<wend>_<model_stem>)")
    args = p.parse_args()

    if args.model_list:
        extra = Path(args.model_list).read_text().splitlines()
        args.model += [l.strip() for l in extra if l.strip()]
    if not args.model:
        p.error("at least one model is required via --model or --model-list")

    # ── MPI setup (optional) ─────────────────────────────────────────────────
    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _rank, _size = 0, 1

    # ── resolve model paths and line lists ────────────────────────────────────
    model_paths = [Path(m).resolve() for m in args.model]
    for mp in model_paths:
        if not mp.exists():
            p.error(f"Model file not found: {mp}")

    linelists = ([Path(f).resolve() for f in args.linelist]
                 if args.linelist else [DEFAULT_LINELIST])
    for ll in linelists:
        if not ll.exists():
            p.error(f"Line list not found: {ll}")

    airvac        = "VAC" if args.vac else "AIR"
    vsini_list    = args.vsini
    vmacro_list   = args.vmacro
    broaden_R_list = args.broaden_R

    # ── grid mode (multiple models or multiple broadening values) ─────────────
    is_grid = (len(model_paths) > 1
               or len(vsini_list) > 1
               or len(vmacro_list) > 1
               or len(broaden_R_list) > 1)

    if is_grid:
        def _grid_rundir(mp, vsini, vmacro, broaden_R):
            suffix = ""
            if vsini    > 0: suffix += f"_v{int(vsini)}"
            if vmacro   > 0: suffix += f"_vm{int(vmacro)}"
            if broaden_R > 0: suffix += f"_R{int(broaden_R)}"
            tag = f"synthe_{args.wstart:.0f}-{args.wend:.0f}{suffix}"
            if args.outdir:
                atlas_name = Path(mp).parent.name or Path(mp).stem
                return Path(args.outdir).resolve() / atlas_name / tag
            return Path(mp).parent / tag

        combos = list(itertools.product(
            model_paths, vsini_list, vmacro_list, broaden_R_list))
        n = len(combos)
        if _rank == 0:
            mpi_note = f"  MPI: {_size} rank(s)" if _size > 1 else ""
            print(f"Grid mode: {n} spectrum/a{mpi_note}")

        my_combos = combos[_rank::_size]

        for ci_local, (mp, vsini, vmacro, broaden_R) in enumerate(my_combos):
            ci_global = _rank + ci_local * _size + 1
            rundir  = _grid_rundir(mp, vsini, vmacro, broaden_R)
            outfile = rundir / "spectrum.dat"
            prefix  = (f"[rank {_rank}  {ci_global}/{n}]  "
                       f"{Path(mp).parent.name}  "
                       f"vsini={vsini:.0f}  vmacro={vmacro:.0f}  R={broaden_R:.0f}")

            if outfile.exists() and outfile.stat().st_size > 0:
                print(f"{prefix}  — skipped (spectrum exists)")
                continue

            print(f"\n{'='*65}\n{prefix}\n{'='*65}")
            try:
                _run_one(rundir, mp, args.wstart, args.wend, args.resolu,
                         args.vturb, vsini, vmacro, broaden_R,
                         args.molecules, linelists, airvac, clean=args.clean)
            except SystemExit as exc:
                print(f"  FAILED: {exc}")

        if _size > 1:
            try:
                _comm.Barrier()
            except Exception:
                pass
        if _rank == 0:
            print(f"\nGrid done.")
        return

    # ── single-model mode — unpack lists and run original pipeline ────────────
    model_path = model_paths[0]
    vsini      = vsini_list[0]
    vmacro     = vmacro_list[0]
    broaden_R  = broaden_R_list[0]

    if args.outdir:
        rundir = Path(args.outdir).resolve()
    else:
        rundir = model_path.parent / f"synthe_{args.wstart:.0f}-{args.wend:.0f}"

    print(f"\nSYNTHE run in: {rundir}")
    print(f"  model   : {model_path}")
    print(f"  lambda  : {args.wstart:.1f}–{args.wend:.1f} Å ({airvac})")
    print(f"  R       : {args.resolu:.0f}")

    _run_one(rundir, model_path, args.wstart, args.wend, args.resolu,
             args.vturb, vsini, vmacro, broaden_R,
             args.molecules, linelists, airvac, clean=args.clean)

    print(f"\nDone. Output: {rundir}/spectrum.dat")
    print(f"  Columns: wavelength (Å)  normalized flux  continuum flux")


if __name__ == "__main__":
    main()
