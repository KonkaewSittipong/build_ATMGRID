#!/usr/bin/env python3
"""MPI Stage B — SYNTHE surface-intensity for every model in a grid directory.

Run under mpirun. Each rank takes a round-robin subset of the models found at
<grid-dir>/run_t*/model_final.dat, and for each one runs the SYNTHE chain in
SURFACE INTENSI mode (mu-resolved, no rotate), then writes a per-node
AtmoGridSpectro HDF5 to <specdir>/<node>.spectro.h5. Resume-safe: nodes whose
output already exists are skipped.

Reuses build_spectro_grid.py (which reuses synthe.py); edits neither.

    mpirun -n 28 python spectro_grid_synthe_mpi.py \\
        --grid-dir grid_output --specdir spectro_grid_h5 \\
        --wstart 4000 --wend 8400 --resolu 100000
"""
import argparse
import glob
import sys
import time
from pathlib import Path

HERE = Path("/lustre/MSSP/sittipong/buildmodule/build_ATMGRID")
sys.path.insert(0, str(HERE))

import build_spectro_grid as B    # reuses synthe.py step functions
import synthe as S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", required=True,
                    help="Dir containing run_t*/model_final.dat (ATLAS12 output)")
    ap.add_argument("--specdir", required=True,
                    help="Output dir for per-node <node>.spectro.h5")
    ap.add_argument("--wstart", type=float, required=True)
    ap.add_argument("--wend",   type=float, required=True)
    ap.add_argument("--resolu", type=float, default=100000.0)
    ap.add_argument("--molecules-below", type=float, default=0.0,
                    help="Enable SYNTHE molecular lines for Teff below this K "
                         "(0 = never; e.g. 5000 for cool nodes)")
    ap.add_argument("--keep-bin", action="store_true",
                    help="Keep spectrum.bin (default: delete after h5 conversion)")
    args = ap.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:
        comm = None
        rank, size = 0, 1

    models = sorted(glob.glob(str(Path(args.grid_dir) / "run_t*" / "model_final.dat")))
    if rank == 0:
        print(f"[Stage B] {len(models)} models  ×  {size} ranks  →  {args.specdir}")
        Path(args.specdir).mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    mine = models[rank::size]
    n_done = n_skip = n_fail = 0

    for mp in mine:
        mp = Path(mp).resolve()
        node = mp.parent.name                      # run_tXXXXgYYpZZ
        out  = Path(args.specdir) / f"{node}.spectro.h5"
        if out.exists() and out.stat().st_size > 0:
            n_skip += 1
            continue

        # parse Teff from node name for optional molecular switch
        try:
            teff = float(node.split("t")[1].split("g")[0])
        except (IndexError, ValueError):
            teff = 1e9
        molecules = (args.molecules_below > 0 and teff < args.molecules_below)

        rundir = mp.parent / f"synthe_intensi_{args.wstart:.0f}-{args.wend:.0f}"
        t0 = time.time()
        try:
            binp = B.build_intensity_bin(
                rundir, mp, args.wstart, args.wend, args.resolu,
                0.0, molecules, [S.DEFAULT_LINELIST], "AIR")
            B.write_spectro_h5(binp, out)
            if not args.keep_bin:
                binp.unlink(missing_ok=True)
            n_done += 1
            print(f"[rank {rank}] OK   {node}  ({int(time.time()-t0)}s"
                  f"{', mol' if molecules else ''})", flush=True)
        except SystemExit as exc:
            n_fail += 1
            print(f"[rank {rank}] FAIL {node}: {exc}", flush=True)
        except Exception as exc:                   # noqa: BLE001
            n_fail += 1
            print(f"[rank {rank}] ERR  {node}: {type(exc).__name__}: {exc}", flush=True)

    print(f"[rank {rank}] done — {n_done} built, {n_skip} skipped, {n_fail} failed",
          flush=True)
    if comm is not None:
        comm.Barrier()
        tb = comm.reduce(n_done, op=MPI.SUM, root=0)
        ts = comm.reduce(n_skip, op=MPI.SUM, root=0)
        tf = comm.reduce(n_fail, op=MPI.SUM, root=0)
        if rank == 0:
            print(f"\n[Stage B] TOTAL — {tb} built, {ts} skipped, {tf} failed")


if __name__ == "__main__":
    main()
