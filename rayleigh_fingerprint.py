"""
rayleigh_fingerprint.py
=======================
Detects duplicate OTDR traces using Rayleigh backscatter fingerprinting.

STRATEGY
--------
Instead of comparing the full backscatter trace (which includes event artifacts),
this module:

1. Parses each SOR file to get the raw trace and detected events (splices,
   connectors, reflections).
2. **Excludes** a buffer zone around every event to remove splice bumps,
   dead-zone artifacts, and connector reflections.
3. Extracts the **pure Rayleigh backscatter segments** between events.
4. Removes the local attenuation slope from each segment to isolate the
   Rayleigh scattering residual — the fiber's unique "fingerprint".
5. Compares fingerprints between traces using length-weighted segment RMSE.

PHYSICAL RATIONALE
------------------
Rayleigh backscatter arises from microscopic density fluctuations frozen into
the glass during fiber drawing.  These fluctuations are deterministic — two
measurements of the same fiber under the same conditions produce identical
Rayleigh patterns.  By isolating the backscatter *between* events, we remove
the variable contribution of splices/connectors and compare what is truly
unique to each fiber.

Because events are excluded, the fingerprint RMSE for true duplicates should
be even lower (closer to the instrument noise floor) than the full-trace
residual RMSE used in sor_residual_rmse324802.py.

USAGE
-----
    # Compare all SOR files in a directory
    python rayleigh_fingerprint.py /path/to/sor/files/

    # Explicit file list
    python rayleigh_fingerprint.py file1.sor file2.sor file3.sor

    # Custom thresholds and buffer
    python rayleigh_fingerprint.py /path/to/files/ --tier1 0.006 --tier2 0.030 --buffer 20

    # Save results to CSV
    python rayleigh_fingerprint.py /path/to/files/ --csv results.csv

    # Verbose: show per-file segment info
    python rayleigh_fingerprint.py /path/to/files/ --verbose

REQUIREMENTS
------------
    pip install numpy scipy
"""

import os
import sys
import argparse
import base64
import csv
import json
import subprocess
import webbrowser
from datetime import datetime
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Import SOR parsers (co-located in same directory) ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sor_residual_rmse324802 import parse_sor, SORTrace

try:
    from sor_reader324741a import parse_sor_full
except ImportError:
    parse_sor_full = None


# ── Thresholds ──────────────────────────────────────────────────────────────
# Tighter than full-trace RMSE because events are excluded
EXACT_DUP_THRESHOLD   = 0.008   # dB  (Rayleigh-only should be cleaner)
PROBABLE_DUP_THRESHOLD = 0.035   # dB

# Event exclusion buffer (samples on each side of an event)
DEFAULT_EVENT_BUFFER = 20       # ~0.5 m at 0.025 m/sample

# Minimum segment length to be usable
MIN_SEGMENT_SAMPLES = 50

# Minimum total Rayleigh points for a comparison to be considered reliable.
# Default: percentage-based (25% of the shorter trace's Rayleigh points).
# The fixed floor below is an absolute minimum that always applies.
MIN_RAYLEIGH_POINTS = 5000
MIN_RAYLEIGH_PCT    = 0.25      # 25% of min(trace_a, trace_b) Rayleigh points


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RayleighSegment:
    """A single clean Rayleigh backscatter segment between events."""
    start_m:       float           # distance at segment start (m)
    end_m:         float           # distance at segment end (m)
    start_idx:     int             # sample index of segment start
    end_idx:       int             # sample index of segment end
    length_samples: int
    slope_dB_per_km: float         # local attenuation slope for this segment
    residual:      np.ndarray      # slope-corrected Rayleigh residual


@dataclass
class RayleighFingerprint:
    """Complete Rayleigh fingerprint for one SOR trace."""
    fiber_id:       str
    filepath:       str
    dx_m:           float
    wavelength_nm:  float
    num_events:     int            # total events found in trace
    num_segments:   int            # usable segments after exclusion
    total_rayleigh_samples: int    # total samples in all segments
    segments:       list           # list of RayleighSegment
    event_positions_m: list        # event distances for layout comparison


@dataclass
class FingerprintMatch:
    """Result of comparing two Rayleigh fingerprints."""
    fiber_a:            str
    fiber_b:            str
    filepath_a:         str
    filepath_b:         str
    rayleigh_rmse:      float      # length-weighted overall RMSE (dB)
    segment_rmses:      list       # per-segment RMSE values
    segment_lengths:    list       # per-segment sample counts
    n_matched_segments: int        # segments that overlapped
    n_total_rayleigh:   int        # total Rayleigh samples compared
    events_match:       bool       # True if event layouts are compatible
    event_dist_diff:    float      # mean absolute event position difference (m)
    tier:               int        # 0=different, 1=exact dup, 2=probable dup
    tier_label:         str


# ── Segment extraction ──────────────────────────────────────────────────────

def _find_noise_floor_idx(trace: np.ndarray, threshold_dB: float = 58.0) -> int:
    """Find the sample index where the trace hits the noise floor."""
    noise_mask = trace >= threshold_dB
    hits = np.where(noise_mask)[0]
    if len(hits) > 0:
        return int(hits[0]) - 10
    return len(trace) - 10


def _fit_segment_slope(trace_segment: np.ndarray, dx_m: float) -> tuple:
    """
    Fit and remove linear attenuation slope from a single segment.
    Returns (slope_dB_per_km, residual_array).
    """
    n = len(trace_segment)
    x = np.arange(n, dtype=np.float64) * dx_m
    y = trace_segment.astype(np.float64)

    # Robust fit: sigma-clip to exclude any remaining micro-anomalies
    c1 = np.polyfit(x, y, 1)
    res1 = y - np.polyval(c1, x)
    sigma = res1.std()
    if sigma > 0:
        mask = np.abs(res1) < 2.5 * sigma
        if mask.sum() >= 20:
            c2 = np.polyfit(x[mask], y[mask], 1)
        else:
            c2 = c1
    else:
        c2 = c1

    fitted = np.polyval(c2, x)
    residual = (y - fitted).astype(np.float32)
    slope_dB_per_km = c2[0] * 1000.0

    return slope_dB_per_km, residual


def extract_fingerprint(sor: SORTrace,
                        event_buffer: int = DEFAULT_EVENT_BUFFER,
                        min_segment: int = MIN_SEGMENT_SAMPLES,
                        noise_threshold_dB: float = 58.0,
                        launch_skip: int = 8) -> RayleighFingerprint:
    """
    Extract Rayleigh fingerprint from a parsed SOR trace.

    Parameters
    ----------
    sor                 : parsed SORTrace from sor_residual_rmse324802
    event_buffer        : samples to skip on each side of every event
    min_segment         : minimum samples for a usable segment
    noise_threshold_dB  : noise floor cutoff
    launch_skip         : samples to skip at trace start (launch dead zone)

    Returns
    -------
    RayleighFingerprint with clean Rayleigh segments
    """
    trace = sor.trace
    dx_m = sor.dx_m
    n = len(trace)

    # Signal region bounds
    signal_start = launch_skip
    signal_end = _find_noise_floor_idx(trace, noise_threshold_dB)

    if signal_end <= signal_start + min_segment:
        raise ValueError(
            f"Signal region too short for {sor.fiber_id} "
            f"({signal_end - signal_start} samples)"
        )

    # Convert event distances to sample indices
    event_indices = []
    event_positions_m = []
    for evt in sor.interior_events:
        dist_m = evt['dist_m']
        idx = int(round(dist_m / dx_m))
        if signal_start < idx < signal_end:
            event_indices.append(idx)
            event_positions_m.append(dist_m)

    # Sort events by position
    if event_indices:
        sorted_pairs = sorted(zip(event_indices, event_positions_m))
        event_indices = [p[0] for p in sorted_pairs]
        event_positions_m = [p[1] for p in sorted_pairs]

    # Build exclusion zones: each event creates a gap of ±buffer
    # Segment boundaries are: [signal_start, evt1-buf, evt1+buf, evt2-buf, ...]
    boundaries = [signal_start]
    for idx in event_indices:
        boundaries.append(idx - event_buffer)
        boundaries.append(idx + event_buffer)
    boundaries.append(signal_end)

    # Extract segments from pairs of boundaries
    segments = []
    for i in range(0, len(boundaries), 2):
        seg_start = max(boundaries[i], signal_start)
        seg_end = min(boundaries[i + 1], signal_end) if i + 1 < len(boundaries) else signal_end

        if seg_end - seg_start < min_segment:
            continue

        seg_trace = trace[seg_start:seg_end]
        slope, residual = _fit_segment_slope(seg_trace, dx_m)

        segments.append(RayleighSegment(
            start_m        = seg_start * dx_m,
            end_m          = seg_end * dx_m,
            start_idx      = seg_start,
            end_idx        = seg_end,
            length_samples = seg_end - seg_start,
            slope_dB_per_km = round(slope, 5),
            residual       = residual,
        ))

    total_samples = sum(s.length_samples for s in segments)

    return RayleighFingerprint(
        fiber_id              = sor.fiber_id,
        filepath              = sor.filepath,
        dx_m                  = dx_m,
        wavelength_nm         = sor.wavelength_nm,
        num_events            = len(event_indices),
        num_segments          = len(segments),
        total_rayleigh_samples = total_samples,
        segments              = segments,
        event_positions_m     = event_positions_m,
    )


# ── Fingerprint comparison ──────────────────────────────────────────────────

def _segments_overlap(sa: RayleighSegment, sb: RayleighSegment) -> bool:
    """Check if two segments have distance overlap."""
    return sa.start_m < sb.end_m and sb.start_m < sa.end_m


def _compare_segment_pair(sa: RayleighSegment, sb: RayleighSegment,
                          dx_m: float) -> tuple:
    """
    Compare two overlapping segments. Returns (rmse, n_samples).
    Aligns by distance and computes RMSE over the common region.
    """
    # Find common distance range
    common_start_m = max(sa.start_m, sb.start_m)
    common_end_m = min(sa.end_m, sb.end_m)

    if common_end_m <= common_start_m:
        return None, 0

    # Convert to local indices within each segment's residual
    offset_a = int(round((common_start_m - sa.start_m) / dx_m))
    offset_b = int(round((common_start_m - sb.start_m) / dx_m))
    length = int(round((common_end_m - common_start_m) / dx_m))

    ra = sa.residual[offset_a: offset_a + length]
    rb = sb.residual[offset_b: offset_b + length]

    min_len = min(len(ra), len(rb))
    if min_len < 20:
        return None, 0

    ra, rb = ra[:min_len], rb[:min_len]
    rmse = float(np.sqrt(np.mean((ra - rb) ** 2)))
    return rmse, min_len


def compare_fingerprints(
    fp_a: RayleighFingerprint,
    fp_b: RayleighFingerprint,
    tier1_thresh: float = EXACT_DUP_THRESHOLD,
    tier2_thresh: float = PROBABLE_DUP_THRESHOLD,
    min_points: int = MIN_RAYLEIGH_POINTS,
    min_pct: float = MIN_RAYLEIGH_PCT,
) -> FingerprintMatch:
    """
    Compare two Rayleigh fingerprints.

    Matches segments by distance overlap, computes per-segment RMSE,
    and returns a length-weighted overall score.  Pairs with fewer than
    max(min_points, min_pct * shorter_trace) total Rayleigh samples are
    marked as low-confidence and excluded from duplicate classification.
    """
    dx_m = fp_a.dx_m  # assume same resolution

    # Effective minimum points: percentage of shorter trace, floored by absolute min
    shorter = min(fp_a.total_rayleigh_samples, fp_b.total_rayleigh_samples)
    effective_min_pts = max(min_points, int(shorter * min_pct))

    # Compare event layouts
    events_match = True
    event_dist_diff = 0.0
    if fp_a.event_positions_m and fp_b.event_positions_m:
        if len(fp_a.event_positions_m) == len(fp_b.event_positions_m):
            diffs = [abs(a - b) for a, b in zip(fp_a.event_positions_m,
                                                  fp_b.event_positions_m)]
            event_dist_diff = sum(diffs) / len(diffs)
            # If events are more than 1m apart on average, layouts differ
            events_match = event_dist_diff < 1.0
        else:
            events_match = False
            # Approximate diff using available events
            event_dist_diff = float('inf')
    elif fp_a.event_positions_m or fp_b.event_positions_m:
        # One has events, the other doesn't
        events_match = len(fp_a.event_positions_m) == 0 and len(fp_b.event_positions_m) == 0

    # Match and compare segments
    segment_rmses = []
    segment_lengths = []
    used_b = set()

    for sa in fp_a.segments:
        best_rmse = None
        best_len = 0
        best_j = -1
        for j, sb in enumerate(fp_b.segments):
            if j in used_b:
                continue
            if _segments_overlap(sa, sb):
                rmse, n = _compare_segment_pair(sa, sb, dx_m)
                if rmse is not None and (best_rmse is None or n > best_len):
                    best_rmse = rmse
                    best_len = n
                    best_j = j

        if best_rmse is not None:
            segment_rmses.append(best_rmse)
            segment_lengths.append(best_len)
            used_b.add(best_j)

    # Length-weighted overall RMSE
    if segment_lengths:
        weights = np.array(segment_lengths, dtype=np.float64)
        rmses = np.array(segment_rmses, dtype=np.float64)
        overall_rmse = float(np.sqrt(np.sum(weights * rmses**2) / np.sum(weights)))
        total_samples = int(np.sum(weights))
    else:
        overall_rmse = float('inf')
        total_samples = 0

    # Classification — minimum Rayleigh point threshold must be met
    # The Rayleigh RMSE is the primary discriminator. Event layout match is
    # informational — the OTDR may detect different numbers of borderline
    # events on back-to-back measurements of the same fiber, so event mismatch
    # alone should not override a strong Rayleigh match.
    if total_samples < effective_min_pts:
        tier, label = 0, "insufficient data"
    elif overall_rmse <= tier1_thresh:
        tier, label = 1, "EXACT DUPLICATE"
    elif overall_rmse <= tier2_thresh:
        tier, label = 2, "PROBABLE DUPLICATE"
    else:
        tier, label = 0, "different"

    return FingerprintMatch(
        fiber_a            = fp_a.fiber_id,
        fiber_b            = fp_b.fiber_id,
        filepath_a         = fp_a.filepath,
        filepath_b         = fp_b.filepath,
        rayleigh_rmse      = round(overall_rmse, 6),
        segment_rmses      = [round(r, 6) for r in segment_rmses],
        segment_lengths    = segment_lengths,
        n_matched_segments = len(segment_rmses),
        n_total_rayleigh   = total_samples,
        events_match       = events_match,
        event_dist_diff    = round(event_dist_diff, 3) if event_dist_diff != float('inf') else -1.0,
        tier               = tier,
        tier_label         = label,
    )


# ── Batch runner ────────────────────────────────────────────────────────────

def run_batch(filepaths: list,
              tier1_thresh: float = EXACT_DUP_THRESHOLD,
              tier2_thresh: float = PROBABLE_DUP_THRESHOLD,
              event_buffer: int = DEFAULT_EVENT_BUFFER,
              min_points: int = MIN_RAYLEIGH_POINTS,
              min_pct: float = MIN_RAYLEIGH_PCT,
              verbose: bool = False) -> tuple:
    """
    Parse all SOR files, extract Rayleigh fingerprints, and compare all pairs.

    Returns (sorted list of FingerprintMatch, event_meta dict).
    The event_meta dict maps fiber_id -> parse_sor_full result for key events.
    """
    # Parse and fingerprint
    fingerprints = []
    event_meta = {}
    print(f"\nParsing {len(filepaths)} SOR files and extracting Rayleigh fingerprints...")
    for fp in filepaths:
        _rfp = None
        try:
            sor = parse_sor(fp)
            _rfp = extract_fingerprint(sor, event_buffer=event_buffer)
            fingerprints.append(_rfp)
            if verbose:
                print(f"  {_rfp.fiber_id:<20}  "
                      f"events={_rfp.num_events}  "
                      f"segments={_rfp.num_segments}  "
                      f"rayleigh_pts={_rfp.total_rayleigh_samples:,}  "
                      f"λ={_rfp.wavelength_nm:.1f}nm")
        except Exception as e:
            print(f"  WARNING: {os.path.basename(fp)}: {e}")

        # Also parse with event reader for key events panel
        if parse_sor_full is not None:
            try:
                m = parse_sor_full(fp, trim=True)
                if m:
                    # Use the same fiber_id as the fingerprint so keys match
                    key = _rfp.fiber_id if _rfp else (
                        os.path.basename(fp)
                        .replace('_1550.sor', '')
                        .replace('.sor', '')
                        .replace('.SOR', ''))
                    event_meta[key] = m
            except Exception:
                pass

    if len(fingerprints) < 2:
        print("Need at least 2 valid fingerprints to compare.")
        return [], event_meta

    # Pairwise comparison
    n_pairs = len(fingerprints) * (len(fingerprints) - 1) // 2
    print(f"\nComparing {n_pairs} Rayleigh fingerprint pairs...")

    results = []
    for a, b in combinations(fingerprints, 2):
        try:
            m = compare_fingerprints(a, b, tier1_thresh, tier2_thresh,
                                     min_points, min_pct)
            results.append(m)
        except Exception as e:
            print(f"  WARNING: {a.fiber_id} ↔ {b.fiber_id}: {e}")

    results.sort(key=lambda r: r.rayleigh_rmse)

    # Gap-based reclassification
    results = classify_by_gap(results)

    return results, event_meta


# ── Gap-based duplicate classification ─────────────────────────────────────

def classify_by_gap(results, min_gap_ratio=2.0, min_cluster_size=1):
    """
    Reclassify pairs using natural-break gap analysis.

    Walk the sorted RMSE list and find the largest multiplicative gap between
    consecutive reliable pairs.  Everything below that gap is promoted to
    EXACT DUPLICATE (tier 1) if the gap ratio exceeds min_gap_ratio.

    This catches clusters like [6.5, 8.1, 8.3, ... gap ... 30.4] where
    fixed thresholds might miss pairs just above the tier-1 cutoff.

    Parameters
    ----------
    results       : sorted list of FingerprintMatch
    min_gap_ratio : minimum ratio between consecutive pairs to count as a
                    natural break (default 2.0 = next pair is 2x higher)
    min_cluster_size : minimum pairs in a cluster to reclassify (default 1)
    """
    reliable = [r for r in results if r.tier_label != "insufficient data"
                and r.rayleigh_rmse < float('inf')]
    if len(reliable) < 2:
        return results

    # Find the largest gap ratio in the top portion of pairs
    # Only look at the first 5% or 50 pairs (whichever is smaller) to avoid
    # finding gaps deep in the "different" distribution
    search_depth = min(50, max(10, len(reliable) // 20))
    search = reliable[:search_depth]

    best_gap_ratio = 1.0
    best_gap_idx = -1  # index of the first pair AFTER the gap
    for i in range(1, len(search)):
        prev_rmse = search[i - 1].rayleigh_rmse
        curr_rmse = search[i].rayleigh_rmse
        if prev_rmse > 0:
            ratio = curr_rmse / prev_rmse
            if ratio > best_gap_ratio:
                best_gap_ratio = ratio
                best_gap_idx = i

    if best_gap_ratio < min_gap_ratio or best_gap_idx < min_cluster_size:
        return results  # no significant gap found

    # Everything below the gap gets promoted to tier 1 (EXACT DUPLICATE)
    cluster_rmses = [r.rayleigh_rmse for r in reliable[:best_gap_idx]]
    gap_threshold = reliable[best_gap_idx - 1].rayleigh_rmse
    next_above = reliable[best_gap_idx].rayleigh_rmse

    print(f"\n  Gap analysis: found natural break at {gap_threshold*1000:.1f} mdB "
          f"→ {next_above*1000:.1f} mdB ({best_gap_ratio:.1f}x gap)")
    print(f"  Promoting {best_gap_idx} pair(s) below the gap to EXACT DUPLICATE")

    # Reclassify
    for r in results:
        if r.tier_label == "insufficient data":
            continue
        if r.rayleigh_rmse <= gap_threshold:
            r.tier = 1
            r.tier_label = "EXACT DUPLICATE"

    return results


# ── Reporting ───────────────────────────────────────────────────────────────

def print_report(results: list, tier1_thresh: float, tier2_thresh: float,
                 min_points: int = MIN_RAYLEIGH_POINTS):
    """Print formatted Rayleigh fingerprint report."""

    tier1 = [r for r in results if r.tier == 1]
    tier2 = [r for r in results if r.tier == 2]
    insufficient = [r for r in results if r.tier_label == "insufficient data"]
    reliable = [r for r in results if r.tier_label != "insufficient data"]

    all_rmse = [r.rayleigh_rmse for r in reliable if r.rayleigh_rmse < float('inf')]
    mean_rmse = float(np.mean(all_rmse)) if all_rmse else 0
    min_rmse = float(np.min(all_rmse)) if all_rmse else 0

    print(f"\n{'═'*76}")
    print(f"  RAYLEIGH'S FINGERPRINT — DUPLICATE REPORT")
    print(f"  Tier 1 threshold (exact dup):    ≤ {tier1_thresh*1000:.1f} mdB")
    print(f"  Tier 2 threshold (probable dup): ≤ {tier2_thresh*1000:.1f} mdB")
    print(f"  Min Rayleigh points:              {min_points:,}")
    print(f"  Method: Rayleigh backscatter between events (events excluded)")
    print(f"{'═'*76}")

    if tier1:
        print(f"\n  ★ EXACT DUPLICATES ({len(tier1)} pair{'s' if len(tier1)!=1 else ''})"
              f" — Rayleigh fingerprints match at noise floor")
        print(f"  {'─'*72}")
        for r in tier1:
            print(f"  {r.fiber_a} ↔ {r.fiber_b}")
            print(f"    Rayleigh RMSE : {r.rayleigh_rmse*1000:.3f} mdB  "
                  f"({r.n_matched_segments} segments, "
                  f"{r.n_total_rayleigh:,} Rayleigh points)")
            if r.segment_rmses:
                seg_str = ', '.join(f"{s*1000:.2f}" for s in r.segment_rmses)
                print(f"    Per-segment   : [{seg_str}] mdB")
            print(f"    Events match  : {'yes' if r.events_match else 'NO'}"
                  f"  (mean event Δ = {r.event_dist_diff:.1f} m)" if r.event_dist_diff >= 0 else "")
    else:
        print(f"\n  No exact duplicates found (nothing below {tier1_thresh*1000:.1f} mdB).")

    if tier2:
        print(f"\n  ~ PROBABLE DUPLICATES ({len(tier2)} pair{'s' if len(tier2)!=1 else ''})")
        print(f"  {'─'*72}")
        for r in tier2:
            print(f"  {r.fiber_a} ↔ {r.fiber_b}")
            print(f"    Rayleigh RMSE : {r.rayleigh_rmse*1000:.3f} mdB  "
                  f"({r.n_matched_segments} segments, "
                  f"{r.n_total_rayleigh:,} pts)")
            print(f"    Events match  : {'yes' if r.events_match else 'NO'}")
    else:
        print(f"\n  No probable duplicates found.")

    # All reliable pairs ranked (insufficient-data pairs omitted from display)
    print(f"\n  ALL RELIABLE PAIRS RANKED BY RAYLEIGH RMSE")
    print(f"  (pairs with < {min_points:,} Rayleigh points excluded)")
    print(f"  {'─'*72}")
    print(f"  {'Pair':<36} {'Ray RMSE':>10} {'Segs':>5} {'Pts':>8} {'Ev?':>4}  Verdict")
    print(f"  {'─'*72}")
    for r in reliable:
        verdict = ("★ " if r.tier == 1 else "~ " if r.tier == 2 else "  ") + r.tier_label
        ev = "yes" if r.events_match else "no"
        rmse_str = f"{r.rayleigh_rmse*1000:9.3f}m" if r.rayleigh_rmse < float('inf') else "      inf"
        print(f"  {r.fiber_a+' ↔ '+r.fiber_b:<36} "
              f"{rmse_str} "
              f"{r.n_matched_segments:5d} "
              f"{r.n_total_rayleigh:8,} "
              f"{ev:>4}  {verdict}")
    print(f"  {'─'*72}")
    print(f"  (m = mdB = millidecibels)")

    # Summary
    print(f"\n  SUMMARY")
    print(f"  {'─'*72}")
    print(f"  Total pairs:       {len(results):,}")
    print(f"  Reliable pairs:    {len(reliable):,}  (≥ {min_points:,} Rayleigh points)")
    print(f"  Filtered out:      {len(insufficient):,}  (insufficient data)")
    print(f"  Mean RMSE:         {mean_rmse*1000:.2f} mdB  (reliable pairs only)")
    print(f"  Min RMSE:          {min_rmse*1000:.3f} mdB")
    print(f"  Exact dups:        {len(tier1)}")
    print(f"  Probable dups:     {len(tier2)}")
    print()


def save_csv(results: list, path: str):
    """Write Rayleigh fingerprint results to CSV."""
    fieldnames = [
        'fiber_a', 'fiber_b',
        'rayleigh_rmse_dB', 'rayleigh_rmse_mdB',
        'n_matched_segments', 'n_total_rayleigh',
        'events_match', 'event_dist_diff_m',
        'segment_rmses_mdB',
        'tier', 'tier_label',
        'filepath_a', 'filepath_b',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            seg_str = ';'.join(f"{s*1000:.3f}" for s in r.segment_rmses)
            w.writerow({
                'fiber_a':              r.fiber_a,
                'fiber_b':              r.fiber_b,
                'rayleigh_rmse_dB':     r.rayleigh_rmse,
                'rayleigh_rmse_mdB':    round(r.rayleigh_rmse * 1000, 3),
                'n_matched_segments':   r.n_matched_segments,
                'n_total_rayleigh':     r.n_total_rayleigh,
                'events_match':         r.events_match,
                'event_dist_diff_m':    r.event_dist_diff,
                'segment_rmses_mdB':    seg_str,
                'tier':                 r.tier,
                'tier_label':           r.tier_label,
                'filepath_a':           r.filepath_a,
                'filepath_b':           r.filepath_b,
            })
    print(f"  Results saved to: {path}")


# ── Key events helpers ──────────────────────────────────────────────────────

def build_exfo_events(m):
    """Build event list from parse_sor_full result for display."""
    if m is None:
        return []
    evts = []
    cum = 0.0
    prev_d = 0.0
    for e in m.get('events', []):
        cum += e['splice_loss']
        evts.append({
            'num':    e['number'],
            'type':   e['type'],
            'dist':   round(e['dist_km'], 4),
            'span':   round(e['dist_km'] - prev_d, 4),
            'splice': round(e['splice_loss'], 3),
            'refl':   round(e['reflection'], 3),
            'slope':  round(e['slope'], 3),
            'cum':    round(cum, 3),
            'is_refl': e['is_reflective'],
            'is_end':  e['is_end'],
        })
        prev_d = e['dist_km']
    return evts


# ── HTML dashboard generation ──────────────────────────────────────────────

def generate_html(results, event_meta, tier1_thresh, tier2_thresh,
                  min_points, route_name=""):
    """Generate a self-contained HTML dashboard with key events panels."""

    tier1_pairs = [r for r in results if r.tier == 1]
    tier2_pairs = [r for r in results if r.tier == 2]
    reliable = [r for r in results if r.tier_label != "insufficient data"]
    # Top 10 closest pairs (regardless of tier) for event inspection
    top10_closest = reliable[:10]

    all_rmse = [r.rayleigh_rmse for r in reliable if r.rayleigh_rmse < float('inf')]
    mean_rmse = float(np.mean(all_rmse)) if all_rmse else 0
    std_rmse = float(np.std(all_rmse)) if all_rmse else 0
    min_rmse = float(np.min(all_rmse)) if all_rmse else 0
    max_rmse = float(np.max(all_rmse)) if all_rmse else 0

    def _pair_obj(r):
        na, nb = r.fiber_a, r.fiber_b
        ma = event_meta.get(na)
        mb = event_meta.get(nb)
        return {
            'name_a':        na,
            'name_b':        nb,
            'rayleigh_rmse': round(r.rayleigh_rmse * 1000, 3),
            'n_segments':    r.n_matched_segments,
            'n_rayleigh':    r.n_total_rayleigh,
            'events_match':  r.events_match,
            'event_diff':    r.event_dist_diff,
            'tier':          r.tier,
            'tier_label':    r.tier_label,
            'seg_rmses':     [round(s * 1000, 2) for s in r.segment_rmses],
            'events_a':      build_exfo_events(ma),
            'events_b':      build_exfo_events(mb),
            'date_time_a':   ma.get('date_time', 0) if ma else 0,
            'date_time_b':   mb.get('date_time', 0) if mb else 0,
        }

    ranked = []
    for r in reliable:
        ranked.append({
            'a': r.fiber_a, 'b': r.fiber_b,
            'rmse': round(r.rayleigh_rmse * 1000, 3),
            'segs': r.n_matched_segments,
            'pts': r.n_total_rayleigh,
            'ev': r.events_match,
            'tier': r.tier,
            'label': r.tier_label,
        })

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    title = f"Rayleigh's Fingerprint — {route_name}" if route_name else "Rayleigh's Fingerprint"

    data_json = json.dumps({
        'title':        title,
        'tier1Pairs':   [_pair_obj(r) for r in tier1_pairs],
        'tier2Pairs':   [_pair_obj(r) for r in tier2_pairs],
        'top10Closest': [_pair_obj(r) for r in top10_closest],
        'ranked':       ranked,
        'numTraces':    len(event_meta) or '?',
        'numPairs':     len(results),
        'numReliable':  len(reliable),
        'tier1Thresh':  tier1_thresh * 1000,
        'tier2Thresh':  tier2_thresh * 1000,
        'minPoints':    min_points,
        'meanRmse':     round(mean_rmse * 1000, 2),
        'stdRmse':      round(std_rmse * 1000, 2),
        'minRmse':      round(min_rmse * 1000, 3),
        'maxRmse':      round(max_rmse * 1000, 1),
        'generated':    generated,
    })

    return HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', data_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rayleigh's Fingerprint — Duplicate Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8f7f5;color:#2c2c2a;padding:24px;max-width:1200px;margin:0 auto}
@media(prefers-color-scheme:dark){body{background:#1a1a18;color:#e8e6df}}
h1{font-size:20px;font-weight:500;margin-bottom:4px}
h2{font-size:16px;font-weight:500;margin:24px 0 8px 0}
.subtitle{font-size:13px;color:#888;margin-bottom:20px}
.panel{background:#fff;border:1px solid rgba(0,0,0,.1);border-radius:12px;padding:16px 20px;margin-bottom:16px}
@media(prefers-color-scheme:dark){.panel{background:#242422;border-color:rgba(255,255,255,.1)}}
.section-label{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#999;margin-bottom:8px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
.metric{background:#fff;border:1px solid rgba(0,0,0,.08);border-radius:10px;padding:14px 16px}
@media(prefers-color-scheme:dark){.metric{background:#242422;border-color:rgba(255,255,255,.08)}}
.metric-label{font-size:11px;color:#999;margin-bottom:4px}
.metric-value{font-size:22px;font-weight:600;line-height:1.2}
.metric-sub{font-size:11px;color:#999;margin-top:2px}
.dup-banner{background:#FAECE7;border:1px solid #F0997B;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#712B13}
@media(prefers-color-scheme:dark){.dup-banner{background:#4A1B0C;border-color:#993C1D;color:#F5C4B3}}
.dup-banner strong{font-weight:600}
.prob-banner{background:#FFF3CD;border:1px solid #E0C060;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#5D5200}
@media(prefers-color-scheme:dark){.prob-banner{background:#2a2a1a;border-color:#665;color:#d4d0a0}}
.prob-banner strong{font-weight:600}
.no-dup-banner{background:#E1F5EE;border:1px solid #5DCAA5;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#04342C}
@media(prefers-color-scheme:dark){.no-dup-banner{background:#04342C;border-color:#0F6E56;color:#9FE1CB}}
.near-banner{background:#FFFDE7;border:1px solid #E0D060;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#5D5700}
@media(prefers-color-scheme:dark){.near-banner{background:#2a2a1a;border-color:#665;color:#d4d0a0}}
.near-banner strong{font-weight:600}
.ranked-table{width:100%;border-collapse:collapse;font-size:12px;font-family:'SF Mono','Fira Code','Cascadia Code',monospace}
.ranked-table th{text-align:left;padding:6px 10px;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#999;border-bottom:1px solid rgba(0,0,0,.1)}
@media(prefers-color-scheme:dark){.ranked-table th{border-bottom-color:rgba(255,255,255,.1)}}
.ranked-table td{padding:5px 10px;border-bottom:0.5px solid rgba(0,0,0,.04)}
@media(prefers-color-scheme:dark){.ranked-table td{border-bottom-color:rgba(255,255,255,.04)}}
.ranked-table tr.tier1{background:rgba(227,75,74,.08)}
.ranked-table tr.tier2{background:rgba(224,192,96,.08)}
@media(prefers-color-scheme:dark){.ranked-table tr.tier1{background:rgba(227,75,74,.12)}.ranked-table tr.tier2{background:rgba(224,192,96,.10)}}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:600}
.tag-t1{background:rgba(227,75,74,.15);color:#a32d2d}
.tag-t2{background:rgba(224,192,96,.2);color:#7a6800}
@media(prefers-color-scheme:dark){.tag-t1{background:rgba(227,75,74,.25);color:#f09595}.tag-t2{background:rgba(224,192,96,.2);color:#d4d0a0}}
.footer{margin-top:24px;padding-top:12px;border-top:0.5px solid rgba(0,0,0,.08);font-size:11px;color:#999;display:flex;justify-content:space-between}
@media(prefers-color-scheme:dark){.footer{border-top-color:rgba(255,255,255,.08)}}
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="subtitle" id="subtitle"></div>
<div id="metrics"></div>
<div id="content"></div>
<div class="footer">
  <span id="footer-left"></span>
  <span>generated by rayleigh_fingerprint.py</span>
</div>

<script>
var DATA = __DATA_PLACEHOLDER__;
var isDark = matchMedia('(prefers-color-scheme:dark)').matches;
var txtSec = isDark ? '#b4b2a9' : '#888780';
var sep = isDark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.1)';
var content = document.getElementById('content');

document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent =
  DATA.numTraces + ' traces \u2022 ' + DATA.numPairs.toLocaleString() + ' pairs (' +
  DATA.numReliable.toLocaleString() + ' reliable) \u2022 generated ' + DATA.generated;
document.getElementById('footer-left').textContent =
  DATA.numTraces + ' files \u2022 Rayleigh backscatter fingerprinting (events excluded)';

// ── Summary metrics ──
(function() {
  var m = document.getElementById('metrics');
  m.className = 'metrics';
  var cards = [
    { label: 'reliable pairs',      value: DATA.numReliable.toLocaleString(), sub: 'of ' + DATA.numPairs.toLocaleString() + ' total' },
    { label: 'mean Rayleigh RMSE',   value: DATA.meanRmse.toFixed(2) + ' mdB', sub: '\u00b1 ' + DATA.stdRmse.toFixed(2) + ' mdB' },
    { label: 'closest pair',         value: DATA.minRmse.toFixed(3) + ' mdB' },
    { label: 'exact duplicates',     value: DATA.tier1Pairs.length.toString(), sub: '\u2264 ' + DATA.tier1Thresh.toFixed(1) + ' mdB' },
    { label: 'probable duplicates',  value: DATA.tier2Pairs.length.toString(), sub: '\u2264 ' + DATA.tier2Thresh.toFixed(1) + ' mdB' },
    { label: 'max RMSE',            value: DATA.maxRmse.toFixed(1) + ' mdB' },
  ];
  cards.forEach(function(c) {
    var d = document.createElement('div');
    d.className = 'metric';
    d.innerHTML = '<div class="metric-label">' + c.label + '</div>' +
      '<div class="metric-value">' + c.value + '</div>' +
      (c.sub ? '<div class="metric-sub">' + c.sub + '</div>' : '');
    m.appendChild(d);
  });
})();

// ── Helper: side-by-side event tables ──
function buildEventPanel(pair, bannerClass) {
  var a = pair.name_a, b = pair.name_b;
  var banner = document.createElement('div');
  banner.className = bannerClass;
  if (bannerClass === 'dup-banner') {
    banner.innerHTML = '<strong>\u2605 Exact duplicate:</strong> ' + a + ' \u2194 ' + b +
      ' &mdash; Rayleigh RMSE = <strong>' + pair.rayleigh_rmse.toFixed(3) + ' mdB</strong>' +
      ' (' + pair.n_segments + ' segments, ' + pair.n_rayleigh.toLocaleString() + ' Rayleigh pts)';
  } else if (bannerClass === 'prob-banner') {
    banner.innerHTML = '<strong>\u223c Probable duplicate:</strong> ' + a + ' \u2194 ' + b +
      ' &mdash; Rayleigh RMSE = <strong>' + pair.rayleigh_rmse.toFixed(3) + ' mdB</strong>' +
      ' (' + pair.n_segments + ' segments, ' + pair.n_rayleigh.toLocaleString() + ' pts)';
  } else {
    banner.innerHTML = '<strong>Near miss:</strong> ' + a + ' \u2194 ' + b +
      ' &mdash; Rayleigh RMSE = ' + pair.rayleigh_rmse.toFixed(3) + ' mdB' +
      ' (' + pair.n_rayleigh.toLocaleString() + ' pts)';
  }
  content.appendChild(banner);

  // Per-segment RMSE breakdown
  if (pair.seg_rmses && pair.seg_rmses.length > 0) {
    var segDiv = document.createElement('div');
    segDiv.style.cssText = 'font-size:11px;color:#888;margin:-8px 0 12px 16px';
    segDiv.textContent = 'Per-segment RMSE: [' + pair.seg_rmses.map(function(s) { return s.toFixed(2); }).join(', ') + '] mdB';
    if (!pair.events_match) segDiv.textContent += ' \u2022 Event layouts differ';
    content.appendChild(segDiv);
  }

  // Side-by-side event tables
  var sec = document.createElement('div');
  sec.className = 'panel';
  sec.innerHTML = '<div class="section-label">Key events: ' + a + ' \u2194 ' + b + '</div>';

  var row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:20px;align-items:flex-start';

  [{ nm: a, evts: pair.events_a, color: '#3266ad', ts: pair.date_time_a },
   { nm: b, evts: pair.events_b, color: '#E24B4A', ts: pair.date_time_b }].forEach(function(side) {
    var col = document.createElement('div');
    col.style.cssText = 'flex:1;min-width:0;overflow-x:auto';
    var tsStr = '';
    if (side.ts) { var d = new Date(side.ts * 1000); tsStr = ' <span style="font-weight:400;font-size:11px;color:' + txtSec + '">' + d.toLocaleString() + '</span>'; }
    col.innerHTML = '<div style="font-size:12px;font-weight:600;margin-bottom:6px;color:' + side.color + '">' + side.nm + tsStr + '</div>';

    if (!side.evts || side.evts.length === 0) {
      col.innerHTML += '<div style="font-size:12px;color:#888">No events</div>';
    } else {
      var tbl = '<table style="width:100%;border-collapse:collapse;font-size:11px;font-family:\'Courier New\',monospace">';
      tbl += '<tr style="border-bottom:1px solid ' + sep + '">' +
        '<th style="text-align:left;padding:4px 6px;color:' + txtSec + '">#</th>' +
        '<th style="text-align:left;padding:4px 6px;color:' + txtSec + '">Type</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Dist (km)</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Span (km)</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Splice (dB)</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Refl (dB)</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Atten (dB/km)</th>' +
        '<th style="text-align:right;padding:4px 6px;color:' + txtSec + '">Cum Loss</th>' +
        '</tr>';

      side.evts.forEach(function(e) {
        var bg = e.is_end
          ? (isDark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.03)')
          : e.is_refl
            ? (isDark ? 'rgba(50,102,173,.18)' : 'rgba(50,102,173,.07)')
            : 'transparent';
        var flag = e.is_end ? '\u25a0' : e.is_refl ? '\u25ba' : '\u00a0';
        var reflStr = e.refl !== 0 ? e.refl.toFixed(3) : '';
        tbl += '<tr style="background:' + bg + ';border-bottom:0.5px solid ' + (isDark ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)') + '">' +
          '<td style="padding:3px 6px">' + e.num + '</td>' +
          '<td style="padding:3px 6px">' + flag + ' ' + e.type + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + e.dist.toFixed(3) + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + e.span.toFixed(3) + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + (e.splice >= 0 ? '+' : '') + e.splice.toFixed(3) + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + reflStr + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + e.slope.toFixed(3) + '</td>' +
          '<td style="padding:3px 6px;text-align:right">' + e.cum.toFixed(3) + '</td>' +
          '</tr>';
      });
      tbl += '</table>';
      col.innerHTML += tbl;
    }
    row.appendChild(col);
  });

  sec.appendChild(row);
  content.appendChild(sec);
}

// ── Render tier 1 (exact duplicates) ──
if (DATA.tier1Pairs.length) {
  DATA.tier1Pairs.forEach(function(d) { buildEventPanel(d, 'dup-banner'); });
}

// ── Render tier 2 (probable duplicates) ──
if (DATA.tier2Pairs.length) {
  var h = document.createElement('h2');
  h.textContent = 'Probable duplicates';
  content.appendChild(h);
  DATA.tier2Pairs.forEach(function(d) { buildEventPanel(d, 'prob-banner'); });
}

// ── No duplicates banner ──
if (!DATA.tier1Pairs.length && !DATA.tier2Pairs.length) {
  var div = document.createElement('div');
  div.className = 'no-dup-banner';
  div.textContent = 'No duplicates found. Closest reliable pair: ' + DATA.minRmse.toFixed(3) + ' mdB Rayleigh RMSE.';
  content.appendChild(div);
}

// ── Top 10 closest pairs ──
if (DATA.top10Closest.length) {
  var h2 = document.createElement('h2');
  h2.textContent = 'Top 10 closest pairs — event inspection';
  content.appendChild(h2);
  DATA.top10Closest.forEach(function(d) {
    var cls = d.tier === 1 ? 'dup-banner' : d.tier === 2 ? 'prob-banner' : 'near-banner';
    buildEventPanel(d, cls);
  });
}

// ── All reliable pairs ranked table ──
(function() {
  var h2 = document.createElement('h2');
  h2.textContent = 'All reliable pairs ranked by Rayleigh RMSE';
  content.appendChild(h2);

  var panel = document.createElement('div');
  panel.className = 'panel';
  panel.style.overflowX = 'auto';

  var tbl = '<table class="ranked-table">';
  tbl += '<tr><th></th><th>Pair</th><th style="text-align:right">Rayleigh RMSE (mdB)</th>' +
    '<th style="text-align:right">Segments</th>' +
    '<th style="text-align:right">Rayleigh Pts</th>' +
    '<th style="text-align:center">Events?</th><th>Verdict</th></tr>';

  DATA.ranked.forEach(function(r, i) {
    var cls = r.tier === 1 ? ' class="tier1"' : r.tier === 2 ? ' class="tier2"' : '';
    var tag = r.tier === 1
      ? '<span class="tag tag-t1">\u2605 EXACT</span>'
      : r.tier === 2
        ? '<span class="tag tag-t2">\u223c PROBABLE</span>'
        : '';
    tbl += '<tr' + cls + '>' +
      '<td style="text-align:right;color:#999">' + (i + 1) + '</td>' +
      '<td>' + r.a + ' \u2194 ' + r.b + '</td>' +
      '<td style="text-align:right">' + r.rmse.toFixed(3) + '</td>' +
      '<td style="text-align:right">' + r.segs + '</td>' +
      '<td style="text-align:right">' + r.pts.toLocaleString() + '</td>' +
      '<td style="text-align:center">' + (r.ev ? '\u2713' : '\u2717') + '</td>' +
      '<td>' + tag + '</td></tr>';
  });
  tbl += '</table>';
  panel.innerHTML = tbl;
  content.appendChild(panel);
})();

// ── Threshold reference ──
(function() {
  var panel = document.createElement('div');
  panel.className = 'panel';
  panel.innerHTML = '<div class="section-label">Detection thresholds</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;font-size:13px">' +
    '<div style="padding:10px 14px;border-radius:8px;background:rgba(227,75,74,.08);border:1px solid rgba(227,75,74,.2)">' +
      '<div style="font-weight:600;color:#a32d2d">Tier 1: Exact duplicate</div>' +
      '<div style="color:#888;font-size:12px">\u2264 ' + DATA.tier1Thresh.toFixed(1) + ' mdB Rayleigh RMSE</div>' +
      '<div style="color:#888;font-size:11px;margin-top:4px">Rayleigh backscatter between events matches at instrument noise floor.</div>' +
    '</div>' +
    '<div style="padding:10px 14px;border-radius:8px;background:rgba(224,192,96,.08);border:1px solid rgba(224,192,96,.25)">' +
      '<div style="font-weight:600;color:#7a6800">Tier 2: Probable duplicate</div>' +
      '<div style="color:#888;font-size:12px">\u2264 ' + DATA.tier2Thresh.toFixed(1) + ' mdB Rayleigh RMSE</div>' +
      '<div style="color:#888;font-size:11px;margin-top:4px">Same fiber, different session. Slight launch or averaging differences.</div>' +
    '</div>' +
    '<div style="padding:10px 14px;border-radius:8px;background:rgba(0,0,0,.03);border:1px solid rgba(0,0,0,.08)">' +
      '<div style="font-weight:600">Different fiber</div>' +
      '<div style="color:#888;font-size:12px">&gt; ' + DATA.tier2Thresh.toFixed(1) + ' mdB Rayleigh RMSE</div>' +
      '<div style="color:#888;font-size:11px;margin-top:4px">Rayleigh scattering patterns differ — distinct fiber measurements.</div>' +
    '</div>' +
    '</div>';
  content.appendChild(panel);
})();
</script>
</body>
</html>"""


# ── PDF export ─────────────────────────────────────────────────────────────

def _evt_row_html(e):
    """Build one HTML table row for an event."""
    bg = '#f0f4fa' if e['is_refl'] and not e['is_end'] else '#f5f5f3' if e['is_end'] else ''
    flag = '\u25a0' if e['is_end'] else '\u25ba' if e['is_refl'] else '&nbsp;'
    refl = f"{e['refl']:.3f}" if e['refl'] != 0 else ''
    sign = '+' if e['splice'] >= 0 else ''
    style = f' style="background:{bg}"' if bg else ''
    return (f'<tr{style}>'
            f'<td>{e["num"]}</td>'
            f'<td>{flag} {e["type"]}</td>'
            f'<td class="r">{e["dist"]:.3f}</td>'
            f'<td class="r">{e["span"]:.3f}</td>'
            f'<td class="r">{sign}{e["splice"]:.3f}</td>'
            f'<td class="r">{refl}</td>'
            f'<td class="r">{e["slope"]:.3f}</td>'
            f'<td class="r">{e["cum"]:.3f}</td>'
            f'</tr>')


def _evt_table_html(name, events, color, date_time=0):
    """Build a full event table HTML block for PDF."""
    ts_str = ''
    if date_time:
        try:
            ts_str = f' <span style="font-weight:400;font-size:9px;color:#888">{datetime.fromtimestamp(date_time).strftime("%Y-%m-%d %H:%M:%S")}</span>'
        except Exception:
            pass
    hdr = (f'<div class="evt-name" style="color:{color}">{name}{ts_str}</div>'
           '<table class="evt"><tr>'
           '<th>#</th><th>Type</th>'
           '<th class="r">Dist (km)</th><th class="r">Span (km)</th>'
           '<th class="r">Splice (dB)</th><th class="r">Refl (dB)</th>'
           '<th class="r">Atten (dB/km)</th><th class="r">Cum Loss</th>'
           '</tr>')
    rows = ''.join(_evt_row_html(e) for e in events)
    return hdr + rows + '</table>'


def generate_pdf_html(pairs_to_show, event_meta, tier1_thresh, tier2_thresh,
                      num_traces, num_pairs, route_name="",
                      chart_png_path=None):
    """Build lightweight print-friendly HTML with chart + event panels for given pairs."""

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    title = f"Rayleigh's Fingerprint — {route_name}" if route_name else "Rayleigh's Fingerprint"

    pair_blocks = []
    for idx, r in enumerate(pairs_to_show):
        na, nb = r.fiber_a, r.fiber_b
        ma = event_meta.get(na)
        mb = event_meta.get(nb)
        evts_a = build_exfo_events(ma)
        evts_b = build_exfo_events(mb)

        tier_cls = 'dup' if r.tier == 1 else 'prob' if r.tier == 2 else 'close'
        tier_icon = '\u2605' if r.tier == 1 else '\u223c' if r.tier == 2 else '#' + str(idx + 1)

        seg_str = ', '.join(f"{s*1000:.2f}" for s in r.segment_rmses)
        stats = f'''
        <div class="banner {tier_cls}">
          <strong>{tier_icon} {r.tier_label}:</strong> {na} \u2194 {nb}
        </div>
        <table class="stats">
          <tr>
            <th>Fiber A</th><th>Fiber B</th>
            <th>Rayleigh RMSE (dB)</th><th>Rayleigh RMSE (mdB)</th>
            <th>Segments</th><th>Rayleigh Points</th>
          </tr>
          <tr>
            <td>{na}</td><td>{nb}</td>
            <td>{r.rayleigh_rmse:.6f}</td><td>{r.rayleigh_rmse*1000:.3f}</td>
            <td>{r.n_matched_segments}</td><td>{r.n_total_rayleigh:,}</td>
          </tr>
          <tr>
            <th>Events Match</th><th>Event Dist Diff (m)</th>
            <th>Tier</th><th>Tier Label</th>
            <th colspan="2">Per-segment RMSE (mdB)</th>
          </tr>
          <tr>
            <td>{"yes" if r.events_match else "NO"}</td><td>{r.event_dist_diff:.1f}</td>
            <td>{r.tier}</td><td>{r.tier_label}</td>
            <td colspan="2">[{seg_str}]</td>
          </tr>
        </table>'''

        ts_a = ma.get('date_time', 0) if ma else 0
        ts_b = mb.get('date_time', 0) if mb else 0
        evt_a_html = _evt_table_html(na, evts_a, '#3266ad', ts_a) if evts_a else f'<div class="evt-name" style="color:#3266ad">{na}</div><div class="no-evt">No events</div>'
        evt_b_html = _evt_table_html(nb, evts_b, '#E24B4A', ts_b) if evts_b else f'<div class="evt-name" style="color:#E24B4A">{nb}</div><div class="no-evt">No events</div>'

        events_section = f'''
        <div class="section-label">Key Events</div>
        <div class="evt-row">
          <div class="evt-col">{evt_a_html}</div>
          <div class="evt-col">{evt_b_html}</div>
        </div>'''

        page_break = ' style="page-break-before:always"' if idx > 0 else ''
        pair_blocks.append(f'<div class="pair-block"{page_break}>{stats}{events_section}</div>')

    pairs_html = '\n'.join(pair_blocks)
    n_tier1 = sum(1 for p in pairs_to_show if p.tier == 1)
    n_tier2 = sum(1 for p in pairs_to_show if p.tier == 2)
    n_closest = len(pairs_to_show) - n_tier1 - n_tier2

    # Embed chart PNG as base64 data URI if provided
    chart_html = ''
    if chart_png_path and os.path.exists(chart_png_path):
        with open(chart_png_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        chart_html = (f'<div class="chart-block">'
                      f'<img src="data:image/png;base64,{b64}" '
                      f'style="width:100%;max-width:100%;height:auto;border-radius:6px;'
                      f'border:1px solid #ddd;margin-bottom:16px" />'
                      f'</div>')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} (PDF)</title>
<style>
@page {{ size: landscape; margin: 12mm 10mm; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #2c2c2a; padding: 16px; font-size: 11px; }}
h1 {{ font-size: 16px; font-weight: 500; margin-bottom: 2px; }}
.subtitle {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
.summary {{ font-size: 11px; color: #555; margin-bottom: 14px; }}
.pair-block {{ margin-bottom: 20px; }}
.banner {{ border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; font-size: 12px; }}
.banner.dup {{ background: #FAECE7; border: 1px solid #F0997B; color: #712B13; }}
.banner.prob {{ background: #FFF3CD; border: 1px solid #E0C060; color: #5D5200; }}
.banner.close {{ background: #EEF2F7; border: 1px solid #B0C4DE; color: #2C3E50; }}
.banner strong {{ font-weight: 600; }}
.stats {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 10px;
           font-family: 'Courier New', monospace; }}
.stats th {{ background: #f4f3f0; padding: 4px 8px; text-align: left; font-weight: 600;
             border: 0.5px solid #ddd; font-size: 9px; color: #555; }}
.stats td {{ padding: 4px 8px; border: 0.5px solid #ddd; }}
.section-label {{ font-size: 10px; font-weight: 600; letter-spacing: .06em;
                  text-transform: uppercase; color: #999; margin: 10px 0 6px; }}
.evt-row {{ display: flex; gap: 14px; }}
.evt-col {{ flex: 1; min-width: 0; overflow: hidden; }}
.evt-name {{ font-size: 11px; font-weight: 600; margin-bottom: 4px; }}
.evt {{ width: 100%; border-collapse: collapse; font-size: 9px;
        font-family: 'Courier New', monospace; }}
.evt th {{ background: #f4f3f0; padding: 3px 5px; text-align: left; font-weight: 600;
           border-bottom: 1px solid #ccc; font-size: 8px; color: #666; }}
.evt td {{ padding: 2px 5px; border-bottom: 0.5px solid #eee; }}
.evt .r, .evt th.r {{ text-align: right; }}
.no-evt {{ font-size: 10px; color: #888; padding: 4px 0; }}
.footer {{ margin-top: 16px; padding-top: 8px; border-top: 0.5px solid #ddd;
           font-size: 9px; color: #999; display: flex; justify-content: space-between; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">{num_traces} traces &bull; {num_pairs:,} pairs &bull; generated {generated}</div>
<div class="summary">
  Tier 1 (exact dup \u2264 {tier1_thresh*1000:.1f} mdB): <strong>{n_tier1} pair{"s" if n_tier1 != 1 else ""}</strong> &nbsp;|&nbsp;
  Tier 2 (probable dup \u2264 {tier2_thresh*1000:.1f} mdB): <strong>{n_tier2} pair{"s" if n_tier2 != 1 else ""}</strong>
  {f"&nbsp;|&nbsp; Closest non-dup pairs shown: {n_closest}" if n_closest else ""}
  &nbsp;|&nbsp; Method: Rayleigh backscatter between events (events excluded)
</div>
{chart_html}
{pairs_html}
<div class="footer">
  <span>{num_traces} files &bull; {len(pairs_to_show)} pairs shown ({n_tier1} exact, {n_tier2} probable, {n_closest} closest) &bull; Rayleigh fingerprinting</span>
  <span>generated by rayleigh_fingerprint.py</span>
</div>
</body>
</html>'''


def _find_chrome():
    """Locate Chrome/Chromium binary."""
    candidates = [
        # macOS
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
        # Linux
        '/usr/bin/google-chrome',
        '/usr/bin/chromium-browser',
    ]
    # Windows paths
    if sys.platform == 'win32':
        candidates = [
            os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe'),
            os.path.expandvars(r'%ProgramFiles%\Microsoft\Edge\Application\msedge.exe'),
        ] + candidates
    return next((p for p in candidates if os.path.exists(p)), None)


def export_pdf(pdf_html_path, pdf_path):
    """Render the PDF-specific HTML to PDF via headless Chrome."""
    chrome = _find_chrome()
    if chrome is None:
        print("  PDF export skipped: Chrome not found.")
        return False

    abs_html = 'file://' + os.path.abspath(pdf_html_path)
    abs_pdf = os.path.abspath(pdf_path)

    try:
        result = subprocess.run(
            [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
             '--run-all-compositor-stages-before-draw',
             '--virtual-time-budget=5000',
             f'--print-to-pdf={abs_pdf}',
             '--print-to-pdf-no-header', '--no-pdf-header-footer',
             abs_html],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(abs_pdf):
            print(f"  PDF written to: {abs_pdf}")
            return True
        else:
            print(f"  PDF export failed (exit {result.returncode}).")
            if result.stderr:
                print(f"  {result.stderr.decode(errors='replace').strip()[:300]}")
            return False
    except Exception as e:
        print(f"  PDF export error: {e}")
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────

def collect_sor_files(inputs: list) -> list:
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
        description="Detect duplicate OTDR traces using Rayleigh backscatter fingerprinting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'inputs', nargs='+',
        help='SOR files or directories containing SOR files'
    )
    parser.add_argument(
        '--tier1', type=float, default=EXACT_DUP_THRESHOLD,
        help=f'Rayleigh RMSE threshold for exact duplicates in dB '
             f'(default: {EXACT_DUP_THRESHOLD})'
    )
    parser.add_argument(
        '--tier2', type=float, default=PROBABLE_DUP_THRESHOLD,
        help=f'Rayleigh RMSE threshold for probable duplicates in dB '
             f'(default: {PROBABLE_DUP_THRESHOLD})'
    )
    parser.add_argument(
        '--buffer', type=int, default=DEFAULT_EVENT_BUFFER,
        help=f'Samples to skip on each side of events '
             f'(default: {DEFAULT_EVENT_BUFFER})'
    )
    parser.add_argument(
        '--min-points', type=int, default=MIN_RAYLEIGH_POINTS,
        help=f'Absolute minimum Rayleigh points for a reliable comparison '
             f'(default: {MIN_RAYLEIGH_POINTS})'
    )
    parser.add_argument(
        '--min-pct', type=float, default=MIN_RAYLEIGH_PCT,
        help=f'Minimum percentage of shorter trace Rayleigh points required '
             f'(default: {MIN_RAYLEIGH_PCT})'
    )
    parser.add_argument(
        '--csv', metavar='PATH',
        help='Save full results table to a CSV file'
    )
    parser.add_argument(
        '--html', metavar='PATH',
        help='Generate HTML dashboard with key events panels'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='Export PDF of flagged pairs (requires --html and Chrome)'
    )
    parser.add_argument(
        '--route-name', default='',
        help='Route name for report title (e.g., "Winterhaven to Niland")'
    )
    parser.add_argument(
        '--chart-png', metavar='PATH',
        help='Distribution chart PNG to embed at top of PDF'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print per-file segment and event info'
    )
    parser.add_argument(
        '--open', action='store_true',
        help='Open HTML report in browser after generation'
    )
    args = parser.parse_args()

    filepaths = collect_sor_files(args.inputs)
    if not filepaths:
        print("No SOR files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(filepaths)} SOR file(s):")
    for fp in filepaths[:10]:
        print(f"  {fp}")
    if len(filepaths) > 10:
        print(f"  ... and {len(filepaths) - 10} more")

    results, event_meta = run_batch(
        filepaths,
        tier1_thresh=args.tier1,
        tier2_thresh=args.tier2,
        event_buffer=args.buffer,
        min_points=args.min_points,
        min_pct=args.min_pct,
        verbose=args.verbose,
    )

    if not results:
        sys.exit(0)

    print_report(results, args.tier1, args.tier2, args.min_points)

    if args.csv:
        save_csv(results, args.csv)

    # HTML dashboard with key events panels
    if args.html:
        print("\nGenerating HTML dashboard ...")
        html = generate_html(results, event_meta, args.tier1, args.tier2,
                             args.min_points, args.route_name)
        with open(args.html, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Dashboard written to: {args.html}")
        print(f"  File size: {len(html):,} bytes")

        if args.open:
            webbrowser.open('file://' + os.path.abspath(args.html))

        # PDF export — always include chart + top 10 closest pairs with events
        if args.pdf:
            reliable = [r for r in results if r.tier_label != "insufficient data"]
            top10 = reliable[:10]
            if top10:
                pdf_html = generate_pdf_html(
                    top10, event_meta, args.tier1, args.tier2,
                    len(event_meta) or len(filepaths), len(results),
                    args.route_name,
                    chart_png_path=args.chart_png)

                pdf_html_path = args.html.replace('.html', '_pdf.html')
                with open(pdf_html_path, 'w', encoding='utf-8') as f:
                    f.write(pdf_html)
                print(f"  PDF source written to: {pdf_html_path}")

                pdf_path = args.html.replace('.html', '.pdf')
                print("Exporting PDF ...")
                export_pdf(pdf_html_path, pdf_path)
            else:
                print("  No reliable pairs to export.")

    return results, event_meta


if __name__ == '__main__':
    main()
