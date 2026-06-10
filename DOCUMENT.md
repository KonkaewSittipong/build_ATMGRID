# build_ATMGRID — Methodology, Physics, Math & Code Reference

This document explains, in depth, how the pipeline turns stellar physics into Icarus
photometry grids: the physics and math of each step, every developer function, the
exact sequence of calls during a run, and how a filter profile is matched to the
synthetic spectrum.

For *how to run it*, see `README.md`. This is the *how/why it works* reference.

---

## 0. The big picture

```
ATLAS12 model atmosphere   →   SYNTHE spectral synthesis   →   band integration
   T(τ), P(τ), ρ(τ) ...          I(μ, λ)  specific intensity       ∫ I·S dλ  →  ⟨f_ν⟩
   (structure of a star)         (emergent light vs angle)         (one number / filter)
```

The product is a grid `flux[Teff, logg, μ]` of band-averaged AB flux densities for a
filter — exactly what Icarus needs to synthesize a binary-star light curve (it sums
the visible stellar surface, weighting each surface element by its `μ` and local
`Teff`/`logg`).

Three nested physical ideas:

1. **Model atmosphere** (ATLAS12): the run of temperature, pressure, density with depth
   for a star of given effective temperature `Teff`, surface gravity `logg`, metallicity.
2. **Specific intensity** `I(μ, λ)` (SYNTHE): how bright the surface is, per wavelength,
   as a function of viewing angle. `μ = cos θ`, where θ is the angle between the line of
   sight and the local surface normal: `μ = 1` = looking straight down (disc centre),
   `μ → 0` = grazing (the limb). Limb darkening = `I` falls as `μ → 0`.
3. **Band flux** (band integration): collapse `I(μ, λ)` over a filter transmission
   `S(λ)` into a single brightness per angle.

---

## 1. Physics & math, step by step

### 1.1 Model atmosphere (ATLAS12)

ATLAS12 solves for a plane-parallel atmosphere in **radiative + convective equilibrium**
and **hydrostatic equilibrium**, given `Teff`, `logg`, and chemical abundances. The
output `model_final.dat` is a table of, per mass-column depth `RHOX`: temperature `T`,
gas pressure `P`, electron number density, opacity, etc. It is computed iteratively:

- **Phase 1 — line selection:** from the ~10⁷-line database, pick the lines that
  matter for this atmosphere's opacity (→ `selected_lines.bin`).
- **Phase 2 — model iteration:** starting from a nearby Kurucz grid model, iterate the
  T/P structure (default 45 iterations) until the emergent flux is constant with depth
  (radiative equilibrium). Convergence is measured by the flux error; the final
  structure is "punched" to `fort.7` → `model_final.dat`.

The **A/F instability** (Teff ≈ 6500–8250 K): in this regime the deep-layer radiative
acceleration can run away, driving the gas pressure to `NaN` and aborting the punch.
The fix is to start from an already-converged neighbour model (see `build_model`).

### 1.2 Specific intensity & the SYNTHE output (spectrv)

SYNTHE takes the converged atmosphere and computes the **emergent specific intensity**
`I(μ, λ)` — the radiation leaving the surface per unit area, solid angle, wavelength,
in each direction `μ`. The card `SURFACE INTENSI N μ₁ … μ_N` requests `N` angles
(SYNTHE's hard limit is `N ≤ 20`, set by `ANGLE(20)` in `spectrv.for`).

**The binary (`spectrv` `fort.7`):** a Fortran *unformatted sequential* file. Each
record is wrapped by a 4-byte length marker on both ends. Layout:

```
header record:  TEFF, GLOG, TITLE[74], WBEGIN, DELTAW, NUMNU, IFSURF, NMU,
                ANGLE[20], NEDGE, WLEDGE[377]
then NUMNU records, one per wavelength, each = 2·NMU doubles:
                Q[0:NMU]   = RESID·SURF = SURFI(μ)   ← the line intensity I_ν(μ)  ← we keep this
                Q[NMU:2NMU]= SURF(μ)                 ← the continuum intensity
```

> Note: this build's `spectrv` writes **2·NMU** values/record (intensity + continuum).
> The stock `icarus-local/Atlas_Reader.py` assumes `NMU`/record and desyncs — which is
> why `build_spectro_grid.read_synthe_intensity` has its own reader that keeps `Q[:NMU]`.

**Units:** `SURFI` is `I_ν` in `erg cm⁻² s⁻¹ Hz⁻¹ sr⁻¹`.

**Wavelength axis (reconstructed, not stored):** SYNTHE samples on a *geometric* grid
of constant resolving power `R = DELTAW`. So with `WBEGIN` (nm) and `R`:

```
λ_i = WBEGIN · (1 + 1/R)^i · 10        (i = 0 … NUMNU-1;  ×10 converts nm → Å)
```

### 1.3 Per-Hz → per-Å (Iν → Iλ) and the log

Filters and Icarus work *per unit wavelength*. Convert with `I_λ dλ = I_ν dν` and
`|dν/dλ| = c/λ²`:

```
I_λ = I_ν · c / λ²          c in Å/s = 2.99792458e18   (= astropy c[m/s] × 1e10)
```

We store **`log(I_λ)`** (unit `dex(erg/(Å cm² s sr))`): the values span orders of
magnitude (disc centre ≫ limb, continuum ≫ line core), and log keeps the dynamic range
tame and interpolation smooth — and matches how Icarus interpolates the grid in μ.

### 1.4 Band integration — matching the filter profile to the spectrum

This is the heart of "how a filter is matched to the spectrum". A filter is a
**transmission curve** `S(λ)` (fraction of light passed vs wavelength; 2-column ASCII
`wavelength_Å  transmission`). The synthetic spectrum is `I_λ(μ, λ)` sampled at the
SYNTHE wavelengths. We want the **band-averaged AB flux density** `⟨f_ν⟩` for each `μ`.

**Step A — put the filter on the spectrum's wavelength grid.** Build an interpolator
`S(λ)` from the filter file (`scipy.interpolate.interp1d`, quadratic, zero outside the
defined range) and evaluate it at the spectrum's `λ_i`. So filter and spectrum now share
the same wavelength sampling — the filter is "matched" onto the spectrum:

```
fb_i = S(λ_i)          # transmission at each spectrum wavelength (0 outside the band)
```

**Step B — integrate.** The AB-system band-averaged flux density (Bessell & Murphy
2012, eq. A12b), with wavelength-space input:

```
            ∫ I_λ(μ,λ) · S(λ) · λ dλ
⟨f_ν⟩(μ) = ───────────────────────────
            ∫ S(λ) · (c·1e10)/λ dλ
```

Numerically (Simpson's rule over the spectrum's `λ` axis):

```python
num = simpson(I_λ * fb * λ, x=λ)            # numerator,  per μ
den = simpson(fb * (c·1e10) / λ, x=λ)       # denominator (filter normalisation)
f_ν = num / den
```

The numerator is the light the filter lets through (intensity × transmission, with the
`λ` from the `f_ν` ↔ `f_λ` Jacobian); the denominator normalises by the filter's own
response so the result is a *mean* flux density, independent of overall filter
throughput. The integral only samples where `S>0`, so the spectrum must span the filter.
Result: `f_ν` in `erg cm⁻² s⁻¹ Hz⁻¹ sr⁻¹`; we store `log(f_ν)`.

**Why this is the whole "matching":** the spectrum carries all the lines and continuum;
the filter `S(λ)` is just the weight function. The band flux is the transmission-weighted
mean of the spectrum — so a redder filter weights the red part of `I_λ`, a bluer one the
blue, and absorption lines inside the band pull the flux down (line blanketing).

### 1.5 Pivot wavelength

A single representative wavelength for the band (Bessell & Murphy 2012, eq. A15):

```
λ_pivot = sqrt( ∫ S·λ dλ / ∫ S/λ dλ )
```

It's the wavelength where `f_ν = f_λ · λ_pivot²/c` is *exact*, so it's the natural label
for an AB grid. Stored in `meta/pivot`; used for plotting and for interpolating `ext`.

### 1.6 The μ axis (17 native → 91 icarus)

SYNTHE gives `I(μ)` at the native 17 angles (≤20 limit). The stock icarus grids use 91
angles (fine near the limb, coarse near disc centre). Because **band integration is
linear in `I`**:

```
interp_μ[ ∫ I(μ,λ)·S dλ ]  =  ∫ interp_μ[ I(μ,λ) ]·S dλ
```

interpolating the *band flux* in μ is identical to interpolating the *intensity* then
integrating — so Stage C linearly interpolates `log⟨f_ν⟩` from the native 17 onto the 91
(`np.interp` per `Teff,logg`). Measured error vs computing all 91 natively: **~1 mmag
mean**, ≤8 mmag only at the extreme limb (negligible flux). `MU_NATIVE=True` instead
computes all 91 directly (SYNTHE in card-safe batches), exact but ~8× the cost.

### 1.7 Extinction metadata (ext, A_V, R_V)

Interstellar dust dims a star by `A_λ` magnitudes (more in the blue). Standard quantities:
`A_V` (dimming in V), `E(B-V) = A_B − A_V` (reddening), `R_V = A_V/E(B-V) ≈ 3.1` (dust
grain property). The grid stores, per band, the extinction-curve value on the
Schlafly & Finkbeiner (2011) scale:

```
ext = (A_band / A_V) / 0.78          # = coeff_SF / R_V / 0.78,  R_V = 3.1
```

For c/o (not in the library) `ext` is interpolated from the stock grids' `ext`-vs-`pivot`
(o=0.846 @ 6827 Å; c=1.228 @ 5368 Å). It's metadata only — it tells Icarus how to redden
the model; it does not change the intensities.

### 1.8 AB zeropoint

The grids are AB: `m = −2.5 log₁₀ f_ν − 48.6`. So `meta/zp = −48.6`, `meta/magsys = "AB"`.

---

## 2. The pipeline: exact call sequence of a run

Command (from a submit script): `mpirun -n N python src/run_grid.py config_c.py`
then serially `python src/run_grid.py config_c.py --assemble-only`.

```
main(argv)
│
├─ load_config("config_c.py")          # populate module globals from the config
│     → reads BASE, MODELS, H5DIR, TEFF/LOGG/FEH, RESOLU, MU, ...
│     → normalises FILTER(S) → FILTERS list of {file,name,desc,ext,out}
│     → WSTART, WEND = union of all filter spans (± WMARGIN)
│
├─ if "--assemble-only" in argv:  assemble();  return        # STAGE C only (serial)
│
├─ MPI: comm, rank, size
├─ combos = product(frange(*TEFF), frange(*LOGG), FEH)       # the grid points
│
├─ STAGE A  (each rank does combos[rank::size])
│     for (Teff,logg,feh): build_model(Teff,logg,feh)
│        └─ extract_starting_model(pick_grid(feh), …)        # nearest Kurucz start
│        └─ run_atlas12.run_phases(…)                        # phase1 + phase2 → model_final.dat
│        └─ convergence_ok(log);  _cleanup_intermediates()
│     barrier
│     FALLBACK rounds (× FALLBACK_ROUNDS):                    # self-heal A/F failures
│        for each still-missing node:
│           for nb in converged_neighbours(…)[:FALLBACK_NEIGHBOURS]:
│              build_model(…, start_model=nb)   → break on success
│        barrier
│
├─ STAGE B  (each rank does models[rank::size])
│     for model_final.dat:  build_spectrum(mp, out)
│        if MU_NATIVE:  build_spectrum_native(mp,out)         # batches of MU_BATCH angles, merged
│        else:                                                # default
│           scratch = mkdtemp(/dev/shm)
│           build_intensity_bin(scratch, mp, WSTART,WEND,RESOLU, …)   # the SYNTHE chain
│           write_spectro_h5(spectrum.bin, out)              # decode → log Iλ → HDF5 (+fsync)
│           rmtree(scratch)
│     barrier
│
└─ STAGE C  (rank 0 only; serial)
      if size==1: assemble()                                  # serial run does it inline
      else: print "run  python src/run_grid.py <config> --assemble-only"   # durable, separate
```

### 2.1 The SYNTHE chain inside `build_intensity_bin`

```
make_synthe_model(model, out.mod, intensi=True, mu_angles)   # prepend "SURFACE INTENSI N …" card
step_xnfpelsyn   → opacity-sampled model       (fort.10 → opacity_model.dat)
step_synbeg      → initialise line tapes        (wavelength range, R, fort.12/14/19/20/93)
step_rgfalllinesnew → read atomic line list     (gfall.dat → line tapes)
[molecules]      → step_rmolec / rschwenk / rh2ofast   (if SYNTHE_MOLECULES)
step_synthe      → compute line-opacity spectrum (the CPU-heavy step; cost ∝ #lines)
step_spectrv     → integrate emergent I(μ)       → fort.7 → spectrum.bin
```

`make_synthe_model` also writes `ITERATIONS 1 / PRINT / PUNCH / CORRECTION OFF /
PRESSURE OFF` cards; without `SURFACE INTENSI`, READIN leaves `IFSURF=0` and spectrv
emits zeros. `rotate/broaden/converfer` are **skipped** — they would collapse the μ axis.

---

## 3. Function reference

### `src/run_grid.py` — the pipeline driver
- **`load_config(path)`** — exec the config `.py`, set module globals. Resolves
  relative paths against `BASE`; normalises single-`FILTER` or `FILTERS`-list into one
  `FILTERS` list; computes `WSTART/WEND` as the **union** of all filter spans ± `WMARGIN`.
- **`frange(start,stop,step)`** — inclusive float range for the grid axes.
- **`node_name(teff,logg,feh)`** — `run_t<T>g<logg*10><p|m><|feh|*10>` (e.g. `run_t5500g40p00`).
- **`converged_neighbours(teff,logg,feh,exclude)`** — converged models (same feh) sorted
  by distance in `((ΔT/250)², (Δlogg/0.5)²)`; the start-model candidates for the fallback.
- **`build_model(teff,logg,feh,start_model=None)`** — make one ATLAS12 model: extract a
  Kurucz start (or copy `start_model`), call `run_atlas12.run_phases`, check convergence,
  clean up. Catches the `SystemExit` that `run_phases` raises if `fort.7` isn't punched.
- **`_scratch_base()`** — first writable node-local scratch dir (`SCRATCH`=/dev/shm, then
  /tmp); never lustre/`$TMPDIR`.
- **`build_spectrum(mp,out)`** — Stage-B worker (interp mode): SYNTHE on /dev/shm scratch
  → `write_spectro_h5`. Dispatches to `build_spectrum_native` if `MU_NATIVE`.
- **`build_spectrum_native(mp,out)`** — compute all `MU` angles natively: SYNTHE in
  card-safe batches of `MU_BATCH`, merge via `write_spectro_h5_merged`.
- **`assemble()`** — Stage C: glob node spectra; load each filter; read each spectrum
  **once** and band-integrate through **all** filters (`band_integrate_AB`); μ-interpolate
  to `MU`; write one photometry grid per filter (+ pivot, ext, fsync).
- **`main()`** — parse `<config> [--assemble-only]`, run the stages (above).

### `src/build_spectro_grid.py` — node worker (model → SYNTHE → spectro h5)
- **`read_synthe_intensity(fn)`** — decode this build's `spectrv` binary (2·NMU/record),
  keep `Q[:NMU]` = `SURFI`; reconstruct the wavelength axis; return a dict
  (`TEFF,GLOG,WBEGIN,DELTAW,NUMNU,NMU,ANGLE,SPECINT,WAV`).
- **`build_model(...)`** — *standalone* single-node model build via `run_atlas12.py`
  subprocess (used by the standalone CLI, not the in-process Stage A).
- **`build_intensity_bin(rundir,mp,wstart,wend,resolu,vturb,molecules,linelists,airvac,mu_angles=None)`**
  — run the 6-step SYNTHE chain (§2.1) → `spectrum.bin`.
- **`write_spectro_h5(bin,out)`** — decode the binary, `I_ν→I_λ`, `log`, sort μ ascending,
  write the `AtmoGridSpectro` HDF5 `(1,1,n_μ,n_wav)` (+ fsync).
- **`write_spectro_h5_merged(bins,out)`** — merge several batch binaries (different μ
  subsets, same wavelengths) into one spectro h5 (used by `build_spectrum_native`).

### `src/band_integrate.py` — filter math (also a standalone CLI)
- **`load_filter(fln,conv,kind)`** — `interp1d` of transmission vs wavelength (Å), 0 outside.
- **`band_integrate_AB(band_func,w,f)`** — the AB band-flux integral (§1.4), Simpson, over
  the last (wavelength) axis. `C_A_PER_S = 2.99792458e18` Å/s.
- **`pivot_wavelength(band_func,w)`** — §1.5.

### `src/synthe.py` — SYNTHE driver (own copy; assets in packages/ATLAS)
- **`make_synthe_model(model,out,molecules,intensi,mu_angles)`** — prepend the control
  cards. `SURFACE INTENSI N μ…` (N=len(mu_angles), default native 17).
- **`_fmt_angle(a)`** — compact Fortran angle string (`.9`, `.0175`, `1.`) so the card
  stays under ~80 columns (else SYNTHE truncates the smallest angles; ≤~12 angles/card).
- **`step_xnfpelsyn / step_synbeg / step_rgfalllinesnew / step_rmolec / step_rschwenk /
  step_rh2ofast / step_synthe / step_spectrv`** — run each SYNTHE executable with the
  right `fort.*` symlinks and stdin cards; rename outputs.
- **`synbeg_card(...)`** — the fixed-format synbeg input (wavelength range nm, R, vturb).

### `src/run_atlas12.py` — ATLAS12 driver (own copy; assets in packages/ATLAS)
- **`pick_grid(feh)`** — choose the Kurucz odfnew starting-grid file for the metallicity.
- **`extract_starting_model(grid,teff,logg,out)`** — pull the closest model block from the
  Kurucz grid as the iteration start.
- **`run_phases(rundir,start,teff,logg,args)`** — phase 1 (line selection) + phase 2
  (model iteration); renames `fort.7`→`model_final.dat`; `sys.exit(1)` if not punched.
- **`convergence_ok(log)`** — parse the flux-error to flag convergence.
- **`run_atlas12(exe,deck,log,cwd)`** — run the `atlas12.exe` with a control deck on stdin.

### `src/mu_icarus.py`
- **`MU_ICARUS`** — the 91 μ angles of the stock icarus grids (e.g. `ATLAS9.AB.Bessell.J.h5`):
  0.01–0.40 in 0.005 steps (fine, limb), 0.40–1.00 in 0.05 steps (coarse, disc centre).

---

## 4. Output HDF5 format (Icarus `AtmoGridPhot`)

```
flux   (n_logtemp, n_logg, n_mu)   = log10⟨f_ν⟩   dex(erg/(Angstrom cm2 s))
cols/  logtemp = ln(Teff)          (dex(K))
       logg                        (dex(m/s2); stored as the ATLAS cgs value)
       mu     = cos(view angle)    (0.01 … 1.0; the icarus 91)
meta/  Z=0.0  magsys="AB"  zp=-48.6  filter=<desc>  pivot=<Å>  ext=<value>
```
The intermediate node spectra add a `wav` axis and store `log I_λ (1,1,n_mu,n_wav)`.

---

## 5. Numerical / engineering notes (the things that bite)

- **`synthe.exe` cost ∝ number of lines.** The blue (dense metal lines + Balmer for hot
  stars) dominates. Mitigation: lower `RESOLU` (band photometry is R-converged by
  ~5000); optionally a trimmed line list (full list kept here for fidelity).
- **Node-local scratch is mandatory.** SYNTHE writes large `fort.*` tapes; on lustre with
  many ranks they thrash the filesystem. We use `/dev/shm` (RAM). `$TMPDIR` here is lustre
  — do not use it.
- **Stage C must be serial / outside mpirun.** Inside mpirun the grid file can land
  0-byte (job exits before lustre syncs). Spectra are `fsync`-ed for the same reason.
- **Card length ≤ ~80 cols** for `SURFACE INTENSI` → ≤ ~12 angles/card (hence batches).
- **Shared `chalawan_cpu` nodes** can starve ranks (CPU contention) and the SYNTHE cost
  is uneven (hot models slowest) → a long tail; prefer a free node / resume-fill stragglers.
- **Resume-safe:** existing `model_final.dat` / `<node>.spectro.h5` are skipped, so a
  cancelled run can be restarted to fill only what's missing.

---

## 6. References

- Kurucz ATLAS12 / SYNTHE (Kurucz 1993; Castelli & Kurucz). `aaatlas12.readme`.
- Bessell & Murphy 2012, PASP 124, 140 — AB band-flux & pivot-wavelength formulae.
- Schlafly & Finkbeiner 2011, ApJ 737, 103 — extinction coefficients (the `ext` values).
- Icarus (R. Breton) — `AtmoGridPhot`/`AtmoGridSpectro` HDF5 format & `Utils/Filter.py`.
