"""
sor_residual_rmse.py
====================
Detects duplicate OTDR traces using slope-corrected backscatter residual RMSE.

STRATEGY
--------
1. Parse each SOR file to extract the raw backscatter trace and measurement params.
2. Fit and remove the bulk attenuation slope (robust linear fit with sigma-clipping
   to exclude splice bumps from the regression).
3. Compare the slope-corrected residuals between every pair of traces using RMSE.
4. Apply a hard threshold based on the instrument noise floor:
     - Residual RMSE < EXACT_DUP_THRESHOLD  →  Tier 1: exact duplicate
       (bumps cancel because splices are at identical positions; RMSE approaches
        the OTDR measurement noise floor of ~0.003–0.005 dB)
     - Residual RMSE < PROBABLE_DUP_THRESHOLD →  Tier 2: probable duplicate
       (same fiber, different measurement session; slight launch/averaging diffs)
     - Everything else  →  different fiber, no action needed

PHYSICAL RATIONALE
------------------
After removing the linear attenuation slope, what remains is a sequence of small
deviations caused by fusion splice events. Two measurements of the same fiber
taken under identical conditions will have their splice bumps at exactly the same
positions, so the bump pattern cancels when residuals are subtracted — the result
approaches instrument noise. Two different fibers have bumps at different positions
that add rather than cancel, producing a residual RMSE floor of ~0.045 dB or higher
regardless of how similar the fibers otherwise appear.

This means residual RMSE is NOT a continuous probability scorer — it's a binary
detection threshold metric. The right question is whether a pair falls below the
physical noise floor of the instrument, not where it ranks in a distribution.

USAGE
-----
    # Single directory — compare all SOR files against each other
    python sor_residual_rmse.py /path/to/sor/files/

    # Explicit file list
    python sor_residual_rmse.py file1.sor file2.sor file3.sor ...

    # With custom thresholds
    python sor_residual_rmse.py /path/to/files/ --tier1 0.008 --tier2 0.040

    # Save results to CSV
    python sor_residual_rmse.py /path/to/files/ --csv results.csv

    # Verbose: print per-file slope info too
    python sor_residual_rmse.py /path/to/files/ --verbose

REQUIREMENTS
------------
    pip install numpy scipy
"""

import struct
import os
import sys
import argparse
import csv
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


# ── Thresholds ────────────────────────────────────────────────────────────────
# Tier 1: at or below OTDR instrument noise floor — exact same measurement session
EXACT_DUP_THRESHOLD   = 0.010   # dB  (conservative; EXFO noise floor ~0.003–0.005)
# Tier 2: same fiber, different session — slight launch / averaging differences
PROBABLE_DUP_THRESHOLD = 0.040   # dB


# ── SOR parser ────────────────────────────────────────────────────────────────

@dataclass
class SORTrace:
    filepath:       str
    fiber_id:       str
    IOR:            float
    dx_m:           float           # meters per sample
    num_points:     int
    wavelength_nm:  float
    averaging_time: int             # seconds
    fiber_length_m: Optional[float]
    trace:          np.ndarray      # raw backscatter (dB), shape (num_points,)
    interior_events: list           # list of dicts with dist_m, loss_dB, code


def _read_cstring(data: bytes, offset: int):
    end = data.index(b'\x00', offset)
    return data[offset:end].decode('latin-1'), end + 1


def parse_sor(filepath: str) -> SORTrace:
    """
    Parse a Telcordia SR-4731 SOR file.
    Extracts IOR, sample spacing, raw trace, fiber ID, and key events.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    IOR = 1.4685          # sensible default if FxdParams not found
    dx_m = 0.025          # 25 m/sample default
    num_points = 0
    wavelength_nm = 0.0
    avg_time = 0
    fiber_id = os.path.splitext(os.path.basename(filepath))[0]

    # ── FxdParams ─────────────────────────────────────────────────────────────
    fxd_pos = data.find(b'FxdParams\x00', 100)
    if fxd_pos >= 0:
        off = fxd_pos + len('FxdParams\x00')
        off += 4                                        # timestamp
        off += 2                                        # distance units
        wl = struct.unpack_from('<H', data, off)[0]; off += 2
        wavelength_nm = wl / 10.0
        off += 8                                        # acq offsets
        num_pw = struct.unpack_from('<H', data, off)[0]; off += 2
        for _ in range(num_pw):
            off += 2                                    # pulse width
            dx_raw = struct.unpack_from('<I', data, off)[0]; off += 4
            npts   = struct.unpack_from('<I', data, off)[0]; off += 4
            dx_m       = dx_raw * 1e-5                 # raw units → metres
            num_points = npts
        IOR_raw = struct.unpack_from('<I', data, off)[0]; off += 4
        IOR = IOR_raw / 100000.0
        off += 2                                        # backscatter coeff
        off += 4                                        # num averages
        avg_time = struct.unpack_from('<H', data, off)[0]

    # ── GenParams ─────────────────────────────────────────────────────────────
    gp_pos = data.find(b'GenParams\x00', 100)
    if gp_pos >= 0:
        off = gp_pos + len('GenParams\x00')
        off += 2                                        # language code (2 bytes, no null)
        off += 1                                        # space separator
        _, off = _read_cstring(data, off)               # cable_id (often blank)
        fid, _  = _read_cstring(data, off)
        fiber_id = fid if fid.strip() else fiber_id

    # ── DataPts ───────────────────────────────────────────────────────────────
    dp_pos = data.find(b'DataPts\x00', 100)
    trace = np.zeros(num_points, dtype=np.float32)
    if dp_pos >= 0:
        off = dp_pos + len('DataPts\x00')
        off += 4                                        # num_points (repeated)
        num_tr = struct.unpack_from('<H', data, off)[0]; off += 2
        if num_tr > 0:
            npts = struct.unpack_from('<I', data, off)[0]; off += 4
            raw  = struct.unpack_from(f'<{npts}H', data, off)
            trace = np.array(raw, dtype=np.float32) / 1000.0  # millidB → dB

    # ── KeyEvents ─────────────────────────────────────────────────────────────
    ke_pos = data.find(b'KeyEvents\x00', 100)
    events = []
    fiber_length_m = None
    if ke_pos >= 0:
        off = ke_pos + len('KeyEvents\x00')
        num_events = struct.unpack_from('<H', data, off)[0]; off += 2
        c_light = 2.998e8   # m/s
        for _ in range(num_events):
            off += 2                                    # event number
            prop_time = struct.unpack_from('<I', data, off)[0]; off += 4
            off += 2                                    # attenuation coeff
            ev_loss = struct.unpack_from('<h', data, off)[0]; off += 2
            off += 4                                    # reflectance
            ev_code = data[off:off+8].decode('latin-1').rstrip('\x00').strip()
            off += 8 + 22                               # code + remaining fields
            dist_m = prop_time * 1e-10 * c_light / (2 * IOR)
            events.append({'dist_m': dist_m, 'loss_dB': ev_loss / 1000.0, 'code': ev_code})
        eof_evts = [e for e in events if '1E' in e['code']]
        if eof_evts:
            fiber_length_m = eof_evts[-1]['dist_m']

    interior = [e for e in events if e['code'].startswith('0F')]

    return SORTrace(
        filepath       = filepath,
        fiber_id       = fiber_id,
        IOR            = IOR,
        dx_m           = dx_m,
        num_points     = num_points,
        wavelength_nm  = wavelength_nm,
        averaging_time = avg_time,
        fiber_length_m = fiber_length_m,
        trace          = trace,
        interior_events= interior,
    )


# ── Attenuation fitting ───────────────────────────────────────────────────────

@dataclass
class SlopeFit:
    slope_dB_per_km: float
    intercept_dB:    float
    residual:        np.ndarray     # slope-corrected signal (dB)
    x_m:             np.ndarray     # distance array for signal region (m)
    start_idx:       int
    end_idx:         int


def fit_attenuation(trace: np.ndarray, dx_m: float,
                    noise_threshold_dB: float = 58.0,
                    launch_skip_samples: int = 8,
                    sigma_clip: float = 2.5) -> SlopeFit:
    """
    Fit and remove the bulk attenuation slope from a backscatter trace.

    Parameters
    ----------
    trace               : raw backscatter array (dB), increasing = more loss
    dx_m                : metres per sample
    noise_threshold_dB  : samples at or above this level are noise floor — excluded
    launch_skip_samples : samples to skip at start (launch dead zone)
    sigma_clip          : sigma threshold for outlier rejection in slope fit
                          (splice bumps are positive outliers — clipped before fitting)

    Returns
    -------
    SlopeFit with slope (dB/km), intercept (dB), and residual array
    """
    n = len(trace)
    dist = np.arange(n, dtype=np.float64) * dx_m

    # Signal region bounds
    noise_mask = trace >= noise_threshold_dB
    first_noise = np.where(noise_mask)[0]
    end_idx = int(first_noise[0]) - 10 if len(first_noise) > 0 else n - 10
    start_idx = launch_skip_samples

    if end_idx <= start_idx + 50:
        raise ValueError(
            f"Signal region too short ({end_idx - start_idx} samples). "
            "Check noise_threshold_dB or launch_skip_samples."
        )

    x = dist[start_idx:end_idx]
    y = trace[start_idx:end_idx].astype(np.float64)

    # Pass 1: initial linear fit
    c1 = np.polyfit(x, y, 1)
    res1 = y - np.polyval(c1, x)

    # Pass 2: sigma-clip outliers (splice bumps appear as positive spikes)
    sigma = res1.std()
    mask = np.abs(res1) < sigma_clip * sigma
    if mask.sum() < 20:
        # If clipping is too aggressive, fall back to unclipped fit
        mask = np.ones(len(x), dtype=bool)

    c2 = np.polyfit(x[mask], y[mask], 1)
    fitted = np.polyval(c2, x)
    residual = (y - fitted).astype(np.float32)

    return SlopeFit(
        slope_dB_per_km = c2[0] * 1000.0,   # dB/m → dB/km
        intercept_dB    = c2[1],
        residual        = residual,
        x_m             = x,
        start_idx       = start_idx,
        end_idx         = end_idx,
    )


# ── Pairwise residual RMSE ────────────────────────────────────────────────────

@dataclass
class PairResult:
    fiber_a:         str
    fiber_b:         str
    filepath_a:      str
    filepath_b:      str
    slope_a:         float          # dB/km
    slope_b:         float          # dB/km
    slope_diff:      float          # |slope_a - slope_b| dB/km
    residual_rmse:   float          # dB — the key metric
    raw_rmse:        float          # dB — for comparison
    ratio:           float          # raw_rmse / residual_rmse
    n_signal_pts:    int            # samples used in common signal region
    tier:            int            # 0=different, 1=exact dup, 2=probable dup
    tier_label:      str


def compute_pair(a: SORTrace, b: SORTrace,
                 tier1_thresh: float = EXACT_DUP_THRESHOLD,
                 tier2_thresh: float = PROBABLE_DUP_THRESHOLD) -> PairResult:
    """
    Compute residual RMSE between two SOR traces.
    Both traces must use the same dx_m (same OTDR resolution setting).
    """
    dx = a.dx_m  # assume both use the same sample spacing

    # Fit slopes
    fa = fit_attenuation(a.trace, dx)
    fb = fit_attenuation(b.trace, dx)

    # Common signal region
    start = max(fa.start_idx, fb.start_idx)
    end   = min(fa.end_idx,   fb.end_idx)

    if end - start < 50:
        raise ValueError(
            f"Common signal region too short for {a.fiber_id} ↔ {b.fiber_id}. "
            "Traces may have very different lengths or noise floors."
        )

    # Residuals in common region
    # Each residual array starts at its own start_idx, so we index accordingly
    offset_a = start - fa.start_idx
    offset_b = start - fb.start_idx
    length   = end - start

    ra = fa.residual[offset_a: offset_a + length]
    rb = fb.residual[offset_b: offset_b + length]

    min_len = min(len(ra), len(rb))
    ra, rb  = ra[:min_len], rb[:min_len]

    res_rmse = float(np.sqrt(np.mean((ra - rb) ** 2)))

    # Raw RMSE (for reference)
    ta = a.trace[:min_len + start]
    tb = b.trace[:min_len + start]
    n  = min(len(ta), len(tb))
    ta_s, tb_s = ta[:n], tb[:n]
    mask = (ta_s < 60) & (tb_s < 60)
    raw_rmse = float(np.sqrt(np.mean((ta_s[mask] - tb_s[mask]) ** 2))) if mask.sum() > 10 else float('inf')

    # Classification
    if res_rmse <= tier1_thresh:
        tier, label = 1, "EXACT DUPLICATE"
    elif res_rmse <= tier2_thresh:
        tier, label = 2, "PROBABLE DUPLICATE"
    else:
        tier, label = 0, "different"

    return PairResult(
        fiber_a       = a.fiber_id,
        fiber_b       = b.fiber_id,
        filepath_a    = a.filepath,
        filepath_b    = b.filepath,
        slope_a       = round(fa.slope_dB_per_km, 5),
        slope_b       = round(fb.slope_dB_per_km, 5),
        slope_diff    = round(abs(fa.slope_dB_per_km - fb.slope_dB_per_km), 5),
        residual_rmse = round(res_rmse, 6),
        raw_rmse      = round(raw_rmse, 4),
        ratio         = round(raw_rmse / res_rmse, 2) if res_rmse > 0 else 0.0,
        n_signal_pts  = min_len,
        tier          = tier,
        tier_label    = label,
    )


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(filepaths: list[str],
              tier1_thresh: float = EXACT_DUP_THRESHOLD,
              tier2_thresh: float = PROBABLE_DUP_THRESHOLD,
              verbose: bool = False) -> list[PairResult]:
    """
    Parse all SOR files and compute pairwise residual RMSE.

    Parameters
    ----------
    filepaths    : list of paths to .sor files
    tier1_thresh : residual RMSE threshold for exact duplicates (dB)
    tier2_thresh : residual RMSE threshold for probable duplicates (dB)
    verbose      : if True, print per-file slope information

    Returns
    -------
    List of PairResult sorted by residual RMSE ascending
    """
    # Parse
    traces = []
    print(f"\nParsing {len(filepaths)} SOR files...")
    for fp in filepaths:
        try:
            t = parse_sor(fp)
            traces.append(t)
            if verbose:
                fa = fit_attenuation(t.trace, t.dx_m)
                print(f"  {t.fiber_id:<20}  IOR={t.IOR:.5f}  "
                      f"dx={t.dx_m:.2f}m  "
                      f"len={t.fiber_length_m/1000:.3f}km  "
                      f"slope={fa.slope_dB_per_km:.4f} dB/km  "
                      f"λ={t.wavelength_nm:.1f}nm")
        except Exception as e:
            print(f"  WARNING: could not parse {fp}: {e}")

    if len(traces) < 2:
        print("Need at least 2 valid traces to compare.")
        return []

    # Pairwise comparison
    n_pairs = len(traces) * (len(traces) - 1) // 2
    print(f"\nComputing {n_pairs} pairwise residual RMSEs...")

    results = []
    for a, b in combinations(traces, 2):
        try:
            r = compute_pair(a, b, tier1_thresh, tier2_thresh)
            results.append(r)
        except Exception as e:
            print(f"  WARNING: {a.fiber_id} ↔ {b.fiber_id}: {e}")

    results.sort(key=lambda r: r.residual_rmse)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[PairResult],
                 tier1_thresh: float,
                 tier2_thresh: float):
    """Print a formatted summary to stdout."""

    tier1 = [r for r in results if r.tier == 1]
    tier2 = [r for r in results if r.tier == 2]

    print(f"\n{'═'*72}")
    print(f"  RESIDUAL RMSE DUPLICATE REPORT")
    print(f"  Tier 1 threshold (exact dup):    ≤ {tier1_thresh*1000:.1f} mdB")
    print(f"  Tier 2 threshold (probable dup): ≤ {tier2_thresh*1000:.1f} mdB")
    print(f"{'═'*72}")

    if tier1:
        print(f"\n  ★ EXACT DUPLICATES ({len(tier1)} pair{'s' if len(tier1)!=1 else ''})"
              f" — residual at instrument noise floor")
        print(f"  {'─'*68}")
        for r in tier1:
            print(f"  {r.fiber_a} ↔ {r.fiber_b}")
            print(f"    Residual RMSE : {r.residual_rmse*1000:.3f} mdB  "
                  f"({r.ratio:.1f}× below raw RMSE of {r.raw_rmse*1000:.1f} mdB)")
            print(f"    Slope A/B     : {r.slope_a:.4f} / {r.slope_b:.4f} dB/km  "
                  f"(diff {r.slope_diff*1000:.2f} mdB/km)")
            print(f"    Signal pts    : {r.n_signal_pts:,}")
    else:
        print(f"\n  No exact duplicates found (nothing below {tier1_thresh*1000:.1f} mdB).")

    if tier2:
        print(f"\n  ~ PROBABLE DUPLICATES ({len(tier2)} pair{'s' if len(tier2)!=1 else ''})"
              f" — same fiber, different session")
        print(f"  {'─'*68}")
        for r in tier2:
            print(f"  {r.fiber_a} ↔ {r.fiber_b}")
            print(f"    Residual RMSE : {r.residual_rmse*1000:.3f} mdB  "
                  f"(raw RMSE {r.raw_rmse*1000:.1f} mdB,  ratio {r.ratio:.1f}×)")
    else:
        print(f"\n  No probable duplicates found.")

    print(f"\n  ALL PAIRS RANKED BY RESIDUAL RMSE")
    print(f"  {'─'*68}")
    print(f"  {'Pair':<36} {'Res RMSE':>10} {'Raw RMSE':>10} {'Ratio':>7}  Verdict")
    print(f"  {'─'*68}")
    for r in results:
        verdict = ("★ " if r.tier == 1 else "~ " if r.tier == 2 else "  ") + r.tier_label
        print(f"  {r.fiber_a+' ↔ '+r.fiber_b:<36} "
              f"{r.residual_rmse*1000:9.3f}m "
              f"{r.raw_rmse*1000:9.1f}m "
              f"{r.ratio:7.1f}x  {verdict}")
    print(f"  {'─'*68}")
    print(f"  (m = mdB = millidecibels)")
    print()


def save_csv(results: list[PairResult], path: str):
    """Write results to a CSV file."""
    fieldnames = [
        'fiber_a', 'fiber_b',
        'residual_rmse_dB', 'residual_rmse_mdB',
        'raw_rmse_dB', 'ratio',
        'slope_a_dB_km', 'slope_b_dB_km', 'slope_diff_dB_km',
        'n_signal_pts', 'tier', 'tier_label',
        'filepath_a', 'filepath_b',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                'fiber_a':              r.fiber_a,
                'fiber_b':              r.fiber_b,
                'residual_rmse_dB':     r.residual_rmse,
                'residual_rmse_mdB':    round(r.residual_rmse * 1000, 3),
                'raw_rmse_dB':          r.raw_rmse,
                'ratio':                r.ratio,
                'slope_a_dB_km':        r.slope_a,
                'slope_b_dB_km':        r.slope_b,
                'slope_diff_dB_km':     r.slope_diff,
                'n_signal_pts':         r.n_signal_pts,
                'tier':                 r.tier,
                'tier_label':           r.tier_label,
                'filepath_a':           r.filepath_a,
                'filepath_b':           r.filepath_b,
            })
    print(f"  Results saved to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_sor_files(inputs: list[str]) -> list[str]:
    """Expand directories to .sor files; pass explicit files through."""
    out = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if fn.lower().endswith('.sor'):
                    out.append(os.path.join(inp, fn))
        elif os.path.isfile(inp):
            out.append(inp)
        else:
            print(f"WARNING: {inp} not found — skipping.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate OTDR traces using slope-corrected residual RMSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'inputs', nargs='+',
        help='SOR files or directories containing SOR files'
    )
    parser.add_argument(
        '--tier1', type=float, default=EXACT_DUP_THRESHOLD,
        help=f'Residual RMSE threshold for exact duplicates in dB '
             f'(default: {EXACT_DUP_THRESHOLD})'
    )
    parser.add_argument(
        '--tier2', type=float, default=PROBABLE_DUP_THRESHOLD,
        help=f'Residual RMSE threshold for probable duplicates in dB '
             f'(default: {PROBABLE_DUP_THRESHOLD})'
    )
    parser.add_argument(
        '--csv', metavar='PATH',
        help='Save full results table to a CSV file'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-file slope and measurement info'
    )
    args = parser.parse_args()

    filepaths = collect_sor_files(args.inputs)
    if not filepaths:
        print("No SOR files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(filepaths)} SOR file(s):")
    for fp in filepaths:
        print(f"  {fp}")

    results = run_batch(
        filepaths,
        tier1_thresh=args.tier1,
        tier2_thresh=args.tier2,
        verbose=args.verbose,
    )

    if not results:
        sys.exit(0)

    print_report(results, args.tier1, args.tier2)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == '__main__':
    main()
