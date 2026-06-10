#!/usr/bin/env python3
"""Stage 2: band-integrate an AtmoGridSpectro HDF5 through a filter transmission
curve to produce an Icarus photometry grid (ATLAS9.AB.<inst>.<band>.h5 format).

Faithfully replicates Icarus' Filter.Band_integration / Filter.Pivot_wavelength
(AB system, wavelength input) and the AtmoGridPhot HDF5 layout, so it runs without
the compiled Icarus package (built for py3.10; this env is py3.11).

  flux_band(logtemp,logg,mu) = ∫ Iλ·S(λ)·λ dλ / ∫ S(λ)·(c·1e10)/λ dλ     (AB)

Input  : spectro grid from build_spectro_grid.py (cols logtemp/logg/mu/wav,
         flux = log Iλ in dex(erg/(Å cm² s sr))).
Filter : 2-column ASCII  wavelength_Å  transmission  (e.g. filters/Misc_Atlas.o.txt).
Output : photometry grid (cols logtemp/logg/mu, flux = log <f_ν>, meta with
         zp/magsys/ext/filter/pivot/Z) — same structure as atmo/ATLAS9.AB.*.h5.

Example
-------
    python band_integrate.py \\
        --spectro spectro_test/ATLAS9_t5500g40p00.spectro.h5 \\
        --filter  filters/Misc_Atlas.o.txt \\
        --filter-name o --filter-desc "ATLAS o" \\
        --out spectro_test/ATLAS9.AB.Atlas.o.h5
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import scipy.integrate
import scipy.interpolate
from astropy.time import Time

# speed of light in Angstrom/s (= astropy c[m/s] * 1e10). Used in the f_nu<->f_lambda
# Jacobian (dν = c/λ² dλ). Keeping c in Å/s makes the band-flux units come out per Hz.
C_A_PER_S = 2.99792458e18


def load_filter(fln, conv=1.0, kind="quadratic"):
    """Build the filter transmission function S(λ) from a 2-col file.

    Returns a callable S(λ): given any wavelengths (Å), it returns the transmission
    there — interpolated (quadratic) between the tabulated points, and 0 outside the
    filter's defined range. This is how the filter is 'matched' onto a spectrum: you
    evaluate S at the spectrum's wavelengths so both share the same sampling.
    """
    w, t = np.loadtxt(fln, unpack=True)[:2]       # col 0 = wavelength, col 1 = transmission
    o = w.argsort()                               # interp1d needs ascending x
    return scipy.interpolate.interp1d(w[o] * conv, t[o], kind=kind,
                                      bounds_error=False, fill_value=0.0)  # 0 beyond the band


def band_integrate_AB(band_func, w, f):
    """AB band-averaged flux density <f_ν> (Bessell & Murphy 2012, eq. A12b).

    band_func: S(λ) from load_filter;  w: wavelengths (Å);  f: Iλ, shape (..., n_wav).
    Integrates over the LAST axis, so any leading axes (e.g. mu) are preserved:
        <f_ν> = ∫ Iλ·S(λ)·λ dλ  /  ∫ S(λ)·(c/λ) dλ
    Numerator = light the filter passes (×λ from the f_ν<->f_λ Jacobian); denominator =
    the filter's own normalisation, so the result is a transmission-weighted MEAN flux
    density in erg/cm²/s/Hz/sr — independent of the filter's overall throughput.
    """
    fb  = band_func(w)                            # S(λ) sampled on the spectrum's grid
    num = scipy.integrate.simpson(f * fb * w, x=w)            # ∫ Iλ S λ dλ  (Simpson over λ)
    den = scipy.integrate.simpson(fb * C_A_PER_S / w, x=w)    # ∫ S (c/λ) dλ
    return num / den                              # <f_ν> (per leading-axis element, e.g. per mu)


def pivot_wavelength(band_func, w):
    """Pivot wavelength of the band (Bessell & Murphy 2012, eq. A15):
        λ_pivot = sqrt( ∫ S·λ dλ / ∫ S/λ dλ ).
    The wavelength where f_ν = f_λ·λ²/c is exact — the natural label for an AB band."""
    fb = band_func(w)
    return np.sqrt(scipy.integrate.simpson(fb * w, x=w)
                   / scipy.integrate.simpson(fb / w, x=w))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spectro", required=True, help="Input AtmoGridSpectro HDF5")
    p.add_argument("--filter",  required=True, help="Filter curve: wavelength_Å transmission")
    p.add_argument("--conv", type=float, default=1.0,
                   help="Multiply filter col1 by this to get Å (default 1.0)")
    p.add_argument("--filter-name", required=True, help="Band name, e.g. o")
    p.add_argument("--filter-desc", default=None, help="Filter description (default: name)")
    p.add_argument("--ext", type=float, default=None,
                   help="Extinction coeff A_band/E(B-V) (Schlafly & Finkbeiner). "
                        "If omitted, stored as NaN (must be set for extinction work).")
    p.add_argument("--out", "-o", required=True, help="Output photometry HDF5")
    args = p.parse_args()

    desc = args.filter_desc or args.filter_name

    # ── read spectro grid ────────────────────────────────────────────────────
    with h5py.File(args.spectro, "r") as h:
        flux_log = h["flux"][...]              # (nT, ng, nmu, nwav) = log Iλ
        logtemp  = h["cols/logtemp"][...]
        logg     = h["cols/logg"][...]
        mu       = h["cols/mu"][...]
        wav      = h["cols/wav"][...]          # Å
        Z        = float(h["meta"].attrs.get("Z", 0.0))

    specint = np.exp(flux_log)                 # Iλ (linear), (nT, ng, nmu, nwav)

    # ── band integration ─────────────────────────────────────────────────────
    bf = load_filter(args.filter, conv=args.conv)

    # sanity: spectrum must span the filter support
    fb = bf(wav)
    nz = wav[fb > 0]
    if nz.size:
        print(f"Filter support: {nz.min():.1f}–{nz.max():.1f} Å  "
              f"(grid covers {wav.min():.1f}–{wav.max():.1f} Å)")
        wfile = np.loadtxt(args.filter)[:, 0] * args.conv
        if wfile.min() < wav.min() or wfile.max() > wav.max():
            print(f"  WARNING: filter ({wfile.min():.1f}–{wfile.max():.1f} Å) extends "
                  f"beyond the grid — band edges will be truncated (transmission set to 0).")

    flux_band = band_integrate_AB(bf, wav, specint)    # (nT, ng, nmu)
    pivot     = float(pivot_wavelength(bf, wav))

    # ── write photometry grid (AtmoGridPhot layout) ──────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with h5py.File(out, "w") as f:
        d = f.create_dataset("flux", data=np.log(flux_band))
        d.attrs["name"] = args.filter_name
        d.attrs["description"] = (
            desc + "-- Specific Intensity Photometry Generated from ATLAS12 on "
            + Time.now().iso)
        d.attrs["unit"] = "dex(erg / (Angstrom cm2 s))"

        m = f.create_dataset("meta", dtype="f")
        m.attrs["zp"]     = -48.6
        m.attrs["magsys"] = "AB"
        m.attrs["ext"]    = (np.nan if args.ext is None else args.ext)
        m.attrs["filter"] = desc
        m.attrs["pivot"]  = pivot
        m.attrs["Z"]      = Z

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

    print(f"\nWrote photometry grid: {out}")
    print(f"  flux (logtemp, logg, mu) = {flux_band.shape}")
    print(f"  pivot = {pivot:.2f} Å   magsys=AB  zp=-48.6  "
          f"ext={'NaN' if args.ext is None else args.ext}")
    if args.ext is None:
        print("  NOTE: ext=NaN — set --ext to the Schlafly & Finkbeiner value "
              "for this band before using the grid for extinction corrections.")


if __name__ == "__main__":
    main()
