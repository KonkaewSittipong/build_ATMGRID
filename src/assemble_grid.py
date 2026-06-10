#!/usr/bin/env python3
"""Stage C — assemble per-node spectro HDF5 files into one Icarus photometry grid.

Reads every <specdir>/run_t*.spectro.h5 (each a single (1,1,n_mu,n_wav) node),
band-integrates them through a filter curve, and stacks the results onto the
(n_logtemp, n_logg, n_mu) grid — the ATLAS9.AB.<inst>.<band>.h5 layout.

Reuses band_integrate.py (filter loading / AB band integration / pivot); no
Icarus import needed.

    python assemble_grid.py \\
        --specdir spectro_grid_h5 \\
        --filter filters/Misc_Atlas.o.txt --filter-name o --filter-desc "ATLAS o" \\
        --ext 0.846 --out grid_output/ATLAS9.AB.Atlas.o.h5
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import h5py
from astropy.time import Time

HERE = Path(__file__).resolve().parent          # src/ (band_integrate lives here)
sys.path.insert(0, str(HERE))
import band_integrate as BI


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specdir", required=True, help="Dir of per-node *.spectro.h5")
    ap.add_argument("--filter",  required=True, help="Filter curve: wavelength_Å transmission")
    ap.add_argument("--conv", type=float, default=1.0)
    ap.add_argument("--filter-name", required=True)
    ap.add_argument("--filter-desc", default=None)
    ap.add_argument("--ext", type=float, default=None,
                    help="Extinction coeff (Icarus convention). For ATLAS o use 0.846.")
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()

    desc  = args.filter_desc or args.filter_name
    files = sorted(glob.glob(str(Path(args.specdir) / "*.spectro.h5")))
    if not files:
        sys.exit(f"No *.spectro.h5 in {args.specdir}")
    print(f"Assembling {len(files)} node spectra through filter '{desc}'")

    # First pass: collect axes (logtemp, logg) and a reference wav/mu.
    nodes = []   # (logtemp, logg, file)
    wav_ref = mu_ref = None
    for f in files:
        with h5py.File(f, "r") as h:
            lt = float(h["cols/logtemp"][0])
            lg = float(h["cols/logg"][0])
            if wav_ref is None:
                wav_ref = h["cols/wav"][...]
                mu_ref  = h["cols/mu"][...]
        nodes.append((lt, lg, f))

    logtemp = np.array(sorted({lt for lt, _, _ in nodes}))
    logg    = np.array(sorted({lg for _, lg, _ in nodes}))
    nT, ng, nmu = len(logtemp), len(logg), len(mu_ref)
    iT = {round(v, 6): k for k, v in enumerate(logtemp)}
    ig = {round(v, 6): k for k, v in enumerate(logg)}
    print(f"  grid: logtemp={nT}  logg={ng}  mu={nmu}  "
          f"(expect {nT*ng} nodes, have {len(files)})")

    band_func = BI.load_filter(args.filter, conv=args.conv)
    pivot = float(BI.pivot_wavelength(band_func, wav_ref))

    flux = np.full((nT, ng, nmu), np.nan)
    nfilled = 0
    for lt, lg, f in nodes:
        with h5py.File(f, "r") as h:
            wav = h["cols/wav"][...]
            spec = np.exp(h["flux"][0, 0])          # (n_mu, n_wav) Iλ
        fb = band_func(wav)
        num = BI.scipy.integrate.simpson(spec * fb * wav, x=wav)
        den = BI.scipy.integrate.simpson(fb * BI.C_A_PER_S / wav, x=wav)
        flux[iT[round(lt, 6)], ig[round(lg, 6)]] = num / den
        nfilled += 1

    missing = int(np.isnan(flux[..., 0]).sum())
    if missing:
        print(f"  WARNING: {missing} of {nT*ng} grid points have no node (left as NaN)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with h5py.File(out, "w") as fh:
        d = fh.create_dataset("flux", data=np.log(flux))
        d.attrs["name"] = args.filter_name
        d.attrs["description"] = (
            desc + "-- Specific Intensity Photometry Generated from ATLAS9 on "
            + Time.now().iso)
        d.attrs["unit"] = "dex(erg / (Angstrom cm2 s))"

        m = fh.create_dataset("meta", dtype="f")
        m.attrs["zp"]     = -48.6
        m.attrs["magsys"] = "AB"
        m.attrs["ext"]    = (np.nan if args.ext is None else args.ext)
        m.attrs["filter"] = desc
        m.attrs["pivot"]  = pivot
        m.attrs["Z"]      = 0.0

        grp = fh.create_group("cols", track_order=True)
        c = grp.create_dataset("logtemp", data=logtemp)
        c.attrs.update(name="logtemp",
                       description="Log (e base) of temperature", unit="dex(K)")
        c = grp.create_dataset("logg", data=logg)
        c.attrs.update(name="logg",
                       description="Log (10 base) of surface gravity",
                       unit="dex(m / s2)")
        c = grp.create_dataset("mu", data=mu_ref)
        c.attrs.update(name="mu", description="cosine of angle of incidence")

    T = np.exp(logtemp)
    print(f"\nWrote {out}")
    print(f"  flux (logtemp, logg, mu) = {flux.shape}   filled {nfilled}/{nT*ng}")
    print(f"  Teff {T.min():.0f}–{T.max():.0f} K   logg {logg.min():.1f}–{logg.max():.1f}")
    print(f"  pivot={pivot:.1f} Å  ext={'NaN' if args.ext is None else args.ext}  AB")


if __name__ == "__main__":
    main()
