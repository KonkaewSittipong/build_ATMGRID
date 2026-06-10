#!/usr/bin/env python3
"""Run ATLAS12 to compute a stellar model atmosphere.

Workflow follows Castelli's Linux procedure:
  Phase 1 — line selection (1 iteration): reads fort.11 (full linelist),
             writes fort.12 (selected lines for this Teff range)
  Phase 2 — model iteration: reads fort.12, iterates to convergence,
             writes fort.7 (final model)

Single model
------------
    python run_atlas12.py --teff 5500 --logg 4.6 --feh -1.0
    python run_atlas12.py --teff 8000 --logg 4.0 --feh  0.0 --iter 45
    python run_atlas12.py --teff 3500 --logg 1.0 --feh -0.5 --molecules
    python run_atlas12.py --teff 50000 --logg 5.68 --feh 0.0 --steps 5
    python run_atlas12.py --teff 50000 --logg 5.68 --feh 0.0 \\
        --start run_t50000g55p00/model_final.dat

Grid (multiple values on any axis)
-----------------------------------
    # 3 × 2 × 2 = 12 models, output in current dir as run_t…g…/ subdirs
    python run_atlas12.py \\
        --teff 5000 5500 6000 \\
        --logg 4.0 4.4 \\
        --feh  0.0 -0.5

    # Same grid under a dedicated output directory
    python run_atlas12.py \\
        --teff 5000 5500 6000 --logg 4.0 4.4 --feh 0.0 -0.5 \\
        --outdir /path/to/my_grid

    # MPI: distribute 12 models over 4 cores
    mpirun -n 4 python run_atlas12.py \\
        --teff 5000 5500 6000 --logg 4.0 4.4 --feh 0.0 -0.5
"""

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
# This driver lives in build_ATMGRID/src/, but the ATLAS12 executable, line lists
# and starting-model grids live in the packages/ATLAS install — point there.
HERE      = Path("/lustre/MSSP/sittipong/packages/ATLAS")
ATLAS12   = HERE / "atlas12" / "atlas12.exe"
LINES_DIR = HERE / "lines"
MODELS    = HERE / "models"

# Metallicity → grid file (odfnew, vturb=2 km/s)
GRID_FILES = {
    -5.0: MODELS / "am50k2odfnew.dat",
    -4.5: MODELS / "am45k2odfnew.dat",
    -4.0: MODELS / "am40k2odfnew.dat",
    -3.5: MODELS / "am35k2odfnew.dat",
    -3.0: MODELS / "am30k2odfnew.dat",
    -2.5: MODELS / "am25k2odfnew.dat",
    -2.0: MODELS / "am20k2odfnew.dat",
    -1.5: MODELS / "am15k2odfnew.dat",
    -1.0: MODELS / "am10k2odfnew.dat",
    -0.5: MODELS / "am05k2odfnew.dat",
     0.0: MODELS / "ap00k2odfnew.dat",
     0.2: MODELS / "ap02k2odfnew.dat",
     0.5: MODELS / "ap05k2odfnew.dat",
     1.0: MODELS / "ap10k2odfnew.dat",
}


def pick_grid(feh):
    available = {k: v for k, v in GRID_FILES.items() if v.exists()}
    if not available:
        sys.exit("ERROR: no grid files found in models/. Download them first.")
    closest = min(available, key=lambda k: abs(k - feh))
    if abs(closest - feh) > 0.01:
        print(f"Note: [M/H]={feh:+.2f} not in grid — using closest: {closest:+.1f}")
    return available[closest]


def extract_starting_model(grid_file, teff, logg, outpath):
    """Pull the closest single model block out of a Kurucz grid file."""
    import re
    header_re = re.compile(r'^TEFF\s+([\d.]+)\s+GRAVITY\s+([\d.]+)')
    lines = open(grid_file).readlines()
    starts = [(i, float(m.group(1)), float(m.group(2)))
              for i, line in enumerate(lines)
              if (m := header_re.match(line))]
    if not starts:
        sys.exit(f"ERROR: no models found in {grid_file}")
    best = min(starts, key=lambda s: ((s[1]-teff)/250)**2 + ((s[2]-logg)/0.5)**2)
    idx  = starts.index(best)
    end  = starts[idx+1][0] if idx+1 < len(starts) else len(lines)
    teff_got, logg_got = best[1], best[2]
    if teff_got != teff or abs(logg_got - logg) > 0.001:
        print(f"  Starting model : Teff={teff_got:.0f} K  logg={logg_got:.2f}"
              f"  (closest to Teff={teff:.0f}, logg={logg:.2f})")
    else:
        print(f"  Starting model : Teff={teff_got:.0f} K  logg={logg_got:.2f}  (exact)")
    open(outpath, 'w').writelines(lines[best[0]:end])


def read_model_params(model_file):
    """Read Teff and logg from the first TEFF/GRAVITY header line of a model file."""
    import re
    header_re = re.compile(r'^TEFF\s+([\d.]+)\s+GRAVITY\s+([\d.]+)')
    with open(model_file) as f:
        for line in f:
            m = header_re.match(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


def symlink(src, dst):
    """Create symlink dst → src, replacing any existing symlink/file."""
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def run_atlas12(exe, input_text, logpath, cwd):
    """Run atlas12.exe with input_text on stdin; log all output."""
    proc = subprocess.Popen(
        [str(exe)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(cwd)
    )
    stdout, _ = proc.communicate(input=input_text)
    Path(logpath).write_text(stdout)
    return proc.returncode, stdout


def convergence_ok(log_text, threshold=0.01):
    """Return (converged, flux_error). Parses the last FLUX ERROR line in the log."""
    import re
    pattern = re.compile(r'flux\s+error[^\d]*(\d[\d.eE+\-]*)', re.IGNORECASE)
    last_err = None
    for line in log_text.splitlines():
        m = pattern.search(line)
        if m:
            try:
                last_err = float(m.group(1))
            except ValueError:
                pass
    if last_err is None:
        return False, None
    return last_err < threshold, last_err


def abundance_scale_cards(abund_file=None):
    """Return ABUNDANCE CHANGE cards from an abundance file.

    The metallicity [M/H] is already baked into the starting model
    (e.g. am10k2odfnew.dat has [M/H]=-1.0 in its ABUND array).
    ABUNDANCE SCALE must NOT be applied again — it would double the offset
    because XABUND = 10^(ABUND + XRELATIVE) and XRELATIVE is already 0
    from the starting model's own ABUNDANCE SCALE 1.0 card.

    Use ABUNDANCE CHANGE only to override specific elements (e.g. alpha
    enhancement, or correcting for different solar reference scales).

    abund_file format (one element per line):
        Z   log_abundance   # optional comment
    Lines starting with # are ignored. Example:
        8   -3.71   # O  (alpha-enhanced at [M/H]=-1.0)
        12  -5.06   # Mg (alpha-enhanced at [M/H]=-1.0)
    """
    if not abund_file:
        return ""
    changes = []
    with open(abund_file) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            z, logA = int(parts[0]), float(parts[1])
            changes.append(f"ABUNDANCE CHANGE {z} {logA:.3f}")
    return "\n".join(changes)


def _cleanup_fort(rundir):
    for f in rundir.glob("fort.*"):
        if f.is_symlink() or f.is_file():
            f.unlink()


def _cleanup_intermediates(rundir):
    """Remove intermediate files after a successful run, keeping only model_final.dat."""
    rundir = Path(rundir)
    for name in ("start_model.dat", "selected_lines.bin",
                 "phase1_linesel.log", "phase2_model.log"):
        p = rundir / name
        if p.exists():
            p.unlink()
    print("  clean: intermediate files removed")


def run_phases(rundir, start_dat, teff, logg, args):
    """Run ATLAS12 phase 1 (line selection) + phase 2 (model iteration).

    Returns (final_model_path, phase2_log_text).
    """
    mol_block = "MOLECULES ON\nREAD MOLECULES\n" if args.molecules else ""
    abund     = abundance_scale_cards(args.abund)
    vturb_cms = args.vturb * 1e5
    conv      = "CONVECTION OVER 1.25 0 36"

    # ── PHASE 1: line selection ───────────────────────────────────────────────
    use_highlines = (teff >= 8000 and (LINES_DIR / "fchighlines.bin").exists())
    print(f"\nPhase 1: line selection (1 iteration"
          f"{', low+high lines' if use_highlines else ', low lines only'})...")

    # Remove any stale fort.12 symlink left by a previous failed phase 2
    _stale = rundir / "fort.12"
    if _stale.is_symlink() or _stale.exists():
        _stale.unlink()

    symlink(start_dat,                     rundir / "fort.3")
    symlink(LINES_DIR / "molecules.dat",   rundir / "fort.2")
    symlink(LINES_DIR / "fclowlines.bin",  rundir / "fort.11")
    if use_highlines:
        symlink(LINES_DIR / "fchighlines.bin", rundir / "fort.21")
    if args.molecules:
        symlink(LINES_DIR / "diatomicsiwl.bin", rundir / "fort.31")
        symlink(LINES_DIR / "tioschwenke.bin",  rundir / "fort.41")
        symlink(LINES_DIR / "h2ofastfix.bin",   rundir / "fort.51")

    deck1 = (
        f"{mol_block}"
        f"READ PUNCH\n"
        f"READ LINES\n"
        f"{conv}\n"
        f"ITERATIONS 1 PRINT 1 PUNCH 0\n"
        + (f"{abund}\n" if abund else "")
        + "BEGIN\n"
          "END\n"
    )

    try:
        rc, out = run_atlas12(ATLAS12, deck1, rundir / "phase1_linesel.log", cwd=rundir)
        if rc != 0:
            print(f"  WARNING: phase 1 returned code {rc}")
            print(f"  Check: {rundir}/phase1_linesel.log")
        else:
            for l in out.splitlines():
                if "LINES FROM" in l or "LINES TOTAL" in l:
                    print(f"  {l.strip()}")
            print("  done.")

        sellin = rundir / "selected_lines.bin"
        fort12 = rundir / "fort.12"
        if fort12.exists() and not fort12.is_symlink() and fort12.stat().st_size > 0:
            fort12.rename(sellin)
            print(f"  Selected lines : {sellin.name}  ({sellin.stat().st_size/1e6:.0f} MB)")
        else:
            sys.exit("ERROR: fort.12 not created in phase 1 — check phase1_linesel.log")
    finally:
        _cleanup_fort(rundir)

    # ── PHASE 2: model iteration ──────────────────────────────────────────────
    print(f"\nPhase 2: model iteration ({args.iter} iterations)...")

    symlink(start_dat,                   rundir / "fort.3")
    symlink(LINES_DIR / "molecules.dat", rundir / "fort.2")
    symlink(sellin,                      rundir / "fort.12")
    if args.molecules:
        symlink(LINES_DIR / "diatomicsiwl.bin", rundir / "fort.31")
        symlink(LINES_DIR / "tioschwenke.bin",  rundir / "fort.41")
        symlink(LINES_DIR / "h2ofastfix.bin",   rundir / "fort.51")
    nltelines = LINES_DIR / "nltelines.bin"
    if nltelines.exists():
        symlink(nltelines, rundir / "fort.19")

    print_flags = " ".join(
        "1" if (i == 1 or i % 15 == 0) else "0" for i in range(1, args.iter + 1)
    )
    punch_flags = " ".join(
        "1" if i == args.iter else "0" for i in range(1, args.iter + 1)
    )

    deck2 = (
        f"{mol_block}"
        f"READ PUNCH\n"
        f"TITLE ATLAS12 Teff={teff:.0f} logg={logg:.2f} FeH={args.feh:+.1f}\n"
        f"OPACITY ON LINES\n"
        f"{conv}\n"
        f"ITERATIONS {args.iter}\n"
        f"PRINT {print_flags}\n"
        f"PUNCH {punch_flags}\n"
        + f"SCALE MODEL 72 -6.875 0.125 {teff:.1f} {logg:.4f}\n"
        f"VTURB {vturb_cms:.2E}\n"
        + (f"{abund}\n" if abund else "")
        + "BEGIN\n"
          "END\n"
    )

    try:
        rc, out2 = run_atlas12(ATLAS12, deck2, rundir / "phase2_model.log", cwd=rundir)
        if rc != 0:
            print(f"  WARNING: phase 2 returned code {rc}")
            print(f"  Check: {rundir}/phase2_model.log")

        fort7 = rundir / "fort.7"
        final = rundir / "model_final.dat"
        if fort7.exists() and fort7.stat().st_size > 0:
            fort7.rename(final)
            print(f"  Output model   : {final.name}")
        else:
            print(f"  WARNING: fort.7 not found — model may not have converged.")
            print(f"  Check: {rundir}/phase2_model.log")
            sys.exit(1)

        log_lines = out2.splitlines()
        flux_errs = [l for l in log_lines if "FLUX ERROR" in l or "flux error" in l.lower()]
        if flux_errs:
            print(f"  {flux_errs[-1].strip()}")
    finally:
        _cleanup_fort(rundir)

    return final, out2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teff",      type=float, nargs="+", required=True,
                   help="Teff value(s) in K — pass multiple for a grid")
    p.add_argument("--logg",      type=float, nargs="+", required=True,
                   help="logg value(s) — pass multiple for a grid")
    p.add_argument("--feh",       type=float, nargs="+", default=[0.0],
                   help="[M/H] value(s) (default 0.0)")
    p.add_argument("--vturb",     type=float, default=2.0,    help="Microturbulence km/s (default 2.0)")
    p.add_argument("--iter",      type=int,   default=45,     help="Iterations phase 2, max 60 (default 45)")
    p.add_argument("--molecules", action="store_true",        help="Include molecular equilibrium")
    p.add_argument("--abund",     type=str,   default=None,   help="Abundance file: Z log_abundance per line")
    p.add_argument("--start",     type=str,   default=None,   help="Use this model file as starting point (skip grid extraction)")
    p.add_argument("--steps",     type=int,   default=1,      help="Steps from starting model to target (default 1 = direct)")
    p.add_argument("--outdir",    type=str,   default=None,   help="Output directory (auto if omitted)")
    p.add_argument("--clean",     action="store_true",        help="Remove intermediate files after each run, keeping only model_final.dat")
    args = p.parse_args()

    if args.iter > 60:
        p.error("--iter max is 60 (atlas12.for: ifprnt(60)/ifpnch(60) arrays)")

    # ── MPI setup (optional) ──────────────────────────────────────────────────
    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _rank, _size = 0, 1

    # ── grid mode: multiple values on any axis, OR --outdir/MPI (avoids
    #    rank collision when multiple ranks share the same outdir) ─────────────
    teff_list = args.teff
    logg_list = args.logg
    feh_list  = args.feh
    if args.outdir or _size > 1 or len(teff_list) > 1 or len(logg_list) > 1 or len(feh_list) > 1:
        base_dir = Path(args.outdir).resolve() if args.outdir else HERE
        combos   = list(itertools.product(teff_list, logg_list, feh_list))
        n        = len(combos)
        if _rank == 0:
            mpi_note = f"  MPI: {_size} rank(s)" if _size > 1 else ""
            print(f"Grid mode: {n} model(s)  →  {base_dir}/{mpi_note}")

        # Distribute combos across MPI ranks (round-robin)
        my_combos = combos[_rank::_size]

        for ci_local, (teff, logg, feh) in enumerate(my_combos):
            ci_global = _rank + ci_local * _size + 1
            sign   = 'p' if feh >= 0 else 'm'
            rundir = base_dir / (f"run_t{int(teff)}g{int(logg*10):02d}"
                                  f"{sign}{int(abs(feh)*10):02d}")
            final  = rundir / "model_final.dat"
            prefix = (f"[rank {_rank}  {ci_global}/{n}]  "
                      f"Teff={teff:.0f}  logg={logg:.2f}  [M/H]={feh:+.1f}")

            if final.exists() and final.stat().st_size > 0:
                print(f"{prefix}  — skipped (model exists)")
                continue

            print(f"\n{'='*65}\n{prefix}\n{'='*65}")
            rundir.mkdir(parents=True, exist_ok=True)

            args.feh = feh   # run_phases reads args.feh for deck title

            start_dat = rundir / "start_model.dat"
            if args.start:
                src = Path(args.start).resolve()
                if not src.exists():
                    print(f"  ERROR: --start not found: {src} — skipped"); continue
                shutil.copy2(src, start_dat)
                print(f"  Starting model : {src}  (user-supplied)")
            else:
                extract_starting_model(pick_grid(feh), teff, logg, start_dat)

            try:
                if args.steps <= 1:
                    final_path, out = run_phases(rundir, start_dat, teff, logg, args)
                    ok, err = convergence_ok(out)
                    if err is not None:
                        print(f"  flux err={err:.4f}  "
                              f"({'converged' if ok else 'NOT converged'})")
                    if args.clean:
                        _cleanup_intermediates(rundir)
                else:
                    # Step-ladder: same logic as single-model mode below
                    start_teff, start_logg = read_model_params(start_dat)
                    if start_teff is None:
                        print("  WARNING: could not read starting params; single step")
                        args.steps = 1
                        final_path, _ = run_phases(rundir, start_dat, teff, logg, args)
                        if args.clean:
                            _cleanup_intermediates(rundir)
                    else:
                        ns = args.steps
                        prev = start_dat
                        for si in range(1, ns + 1):
                            frac = si / ns
                            st   = start_teff + (teff - start_teff) * frac
                            sl   = start_logg + (logg - start_logg) * frac
                            is_last  = (si == ns)
                            step_dir = rundir if is_last else (
                                rundir.parent / (rundir.name + f"_step{si:02d}"))
                            step_dir.mkdir(parents=True, exist_ok=True)
                            step_start = step_dir / "start_model.dat"
                            shutil.copy2(prev, step_start)
                            print(f"\n  Step {si}/{ns}: Teff={st:.0f}  logg={sl:.3f}")
                            final_path, out2 = run_phases(
                                step_dir, step_start, st, sl, args)
                            ok2, err2 = convergence_ok(out2)
                            if err2 is not None:
                                print(f"  flux err={err2:.4f}  "
                                      f"({'converged' if ok2 else 'NOT converged'})")
                            if args.clean:
                                _cleanup_intermediates(step_dir)
                            prev = final_path
            except SystemExit as exc:
                print(f"  FAILED: {exc}")

        if _size > 1:
            try:
                _comm.Barrier()
            except Exception:
                pass
        if _rank == 0:
            print(f"\nGrid done.  Outputs in {base_dir}/")
        return

    # ── single-model mode — unpack lists then fall through to original code ───
    args.teff = teff_list[0]
    args.logg = logg_list[0]
    args.feh  = feh_list[0]

    # ── output directory ──────────────────────────────────────────────────────
    if args.outdir:
        rundir = Path(args.outdir).resolve()
    else:
        sign = 'p' if args.feh >= 0 else 'm'
        rundir = HERE / (f"run_t{int(args.teff)}g{int(args.logg*10):02d}"
                         f"{sign}{int(abs(args.feh)*10):02d}")
    rundir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory : {rundir}/")

    # ── starting model ────────────────────────────────────────────────────────
    start_dat = rundir / "start_model.dat"
    if args.start:
        src = Path(args.start).resolve()
        if not src.exists():
            sys.exit(f"ERROR: --start file not found: {src}")
        shutil.copy2(src, start_dat)
        print(f"  Starting model : {src}  (user-supplied)")
    else:
        grid_file = pick_grid(args.feh)
        extract_starting_model(grid_file, args.teff, args.logg, start_dat)

    # ── single step (default) ─────────────────────────────────────────────────
    if args.steps <= 1:
        final, _ = run_phases(rundir, start_dat, args.teff, args.logg, args)
        if args.clean:
            _cleanup_intermediates(rundir)
        print(f"\nDone.")
        print(f"  Final model : {final}")
        if not args.clean:
            print(f"  Phase 1 log : {rundir}/phase1_linesel.log")
            print(f"  Phase 2 log : {rundir}/phase2_model.log")
        return

    # ── step-ladder ───────────────────────────────────────────────────────────
    # Read Teff/logg baked into the starting model so we can interpolate
    start_teff, start_logg = read_model_params(start_dat)
    if start_teff is None:
        print("WARNING: could not read starting model params; falling back to single step")
        final, _ = run_phases(rundir, start_dat, args.teff, args.logg, args)
        if args.clean:
            _cleanup_intermediates(rundir)
        print(f"\nDone.\n  Final model : {final}")
        return

    n          = args.steps
    prev_model = start_dat
    for i in range(1, n + 1):
        frac      = i / n
        step_teff = start_teff + (args.teff - start_teff) * frac
        step_logg = start_logg + (args.logg - start_logg) * frac
        is_last   = (i == n)
        step_dir  = rundir if is_last else rundir.parent / (rundir.name + f"_step{i:02d}")
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Step {i}/{n}: Teff={step_teff:.0f}  logg={step_logg:.3f}")
        print(f"{'='*60}")

        step_start = step_dir / "start_model.dat"
        shutil.copy2(prev_model, step_start)

        final, out = run_phases(step_dir, step_start, step_teff, step_logg, args)
        ok, err = convergence_ok(out)
        if err is not None:
            status = "converged" if ok else "not converged"
            print(f"  Flux error: {err:.4f}  ({status})")
        if args.clean:
            _cleanup_intermediates(step_dir)
        prev_model = final

    print(f"\nDone (step ladder complete).")
    print(f"  Final model : {prev_model}")
    if not args.clean:
        print(f"  Phase 1 log : {rundir}/phase1_linesel.log")
        print(f"  Phase 2 log : {rundir}/phase2_model.log")


if __name__ == "__main__":
    main()
