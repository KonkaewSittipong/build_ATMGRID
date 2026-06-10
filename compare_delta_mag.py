#!/usr/bin/env python3
"""Compare the two ATLAS photometry grids: Δmag = m_c - m_o (the c-o colour) vs mu.

x-axis = mu (cosine of viewing angle, limb 0.01 -> disc centre 1.0)
y-axis = Δmag = m_c(mu) - m_o(mu)   [magnitudes]

The grids store flux = ln(<f_nu>) and share the same AB zeropoint (-48.6), so the
zeropoint cancels in the difference:

    m_band = -2.5*log10(f_nu) - 48.6 = -(2.5/ln10)*ln(f_nu) - 48.6
    Δmag(mu) = m_c - m_o = -(2.5/ln10) * (flux_c(mu) - flux_o(mu))

This shows the differential limb darkening between the two bands — how the c-o colour
changes from disc centre to limb. One curve per (Teff, logg) requested.

Usage:
    python compare_delta_mag.py                       # default: a few Teff at logg 4.5
    python compare_delta_mag.py --teff 6000 --logg 4.0
    python compare_delta_mag.py --teff 4000 6000 8000 10000 --logg 4.5 --out cmp.png
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")                  # headless: write a PNG, no display needed
import matplotlib.pyplot as plt

BASE  = Path("/lustre/MSSP/sittipong/buildmodule/build_ATMGRID/spectro_grid")
GRID_C = BASE / "ATLAS12.AB.Atlas.c.h5"
GRID_O = BASE / "ATLAS12.AB.Atlas.o.h5"
_2P5_LN10 = 2.5 / np.log(10.0)         # = 1.0857; converts Δln(flux) -> Δmag


def load_grid(path):
    """Return (logtemp, logg, mu, flux) from a photometry grid.
    flux = ln(<f_nu>), shape (nT, ng, nmu)."""
    with h5py.File(path, "r") as h:
        return (h["cols/logtemp"][...], h["cols/logg"][...],
                h["cols/mu"][...], h["flux"][...])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teff", type=float, nargs="+", default=[4000, 6000, 8000, 10000, 12000],
                   help="Teff value(s) K to plot (default: 4000..12000)")
    p.add_argument("--logg", type=float, default=4.5, help="log g (cgs, default 4.5)")
    p.add_argument("--out", default="delta_mag_c_minus_o.png", help="output PNG")
    args = p.parse_args()

    ltc, lgc, muc, fc = load_grid(GRID_C)
    lto, lgo, muo, fo = load_grid(GRID_O)
    # both grids share the same axes (built from the same models); sanity-check mu
    assert np.allclose(muc, muo), "c and o grids have different mu axes"
    mu = muc
    Tc = np.exp(ltc)                   # Teff (K) of the grid's temperature axis
    gg = np.round(lgc, 3)              # logg axis

    def idx(arr, val):                 # nearest grid index to a requested value
        return int(np.argmin(np.abs(arr - val)))

    jg = idx(gg, args.logg)            # logg index (shared)

    fig, ax = plt.subplots(figsize=(7, 5))
    for teff in args.teff:
        iT = idx(Tc, teff)
        # Δmag(mu) = m_c - m_o ; flux is ln(f_nu), AB zeropoints cancel in the difference
        dmag = -_2P5_LN10 * (fc[iT, jg, :] - fo[iT, jg, :])
        if not np.isfinite(dmag).all():            # skip a node with no model (NaN)
            print(f"  skip Teff={Tc[iT]:.0f} logg={gg[jg]:.1f}: NaN (no model)")
            continue
        ax.plot(mu, dmag, marker="o", ms=3.5, ls="none", label=f"{Tc[iT]:.0f} K")
        print(f"  Teff={Tc[iT]:.0f} logg={gg[jg]:.1f}:  Δmag(c-o) "
              f"limb(mu={mu[0]:.2f})={dmag[0]:+.3f}  centre(mu=1)={dmag[-1]:+.3f}  "
              f"span={dmag.max()-dmag.min():.3f} mag")

    ax.set_xlabel("μ = cos(viewing angle)   (limb → disc centre)")
    ax.set_ylabel("Δmag = m$_c$ − m$_o$   (c − o colour)  [mag]")
    ax.set_title(f"ATLAS12 c−o colour vs μ   (log g = {gg[jg]:.1f})")
    ax.legend(title="Teff", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
