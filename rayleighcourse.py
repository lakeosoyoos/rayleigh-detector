"""
rayleighcourse.py
=================
Coarse-resolution duplicate detection for OTDR traces.

When OTDR sample spacing is coarse (e.g. 0.39 m), the standard Rayleigh RMSE
approach cannot separate duplicate fibers from adjacent fibers in the same
cable — the measurement noise floor exceeds the inter-fiber variation.

This module uses a MULTI-METRIC VOTING approach instead:
  1. Extract the slope-corrected Rayleigh residual from each trace
  2. Compute 5 independent shape signatures from the residual
  3. Rank all pairs on each metric independently
  4. Flag pairs that land in the top 0.5% of 4+ metrics simultaneously

A pair that is consistently among the most similar across multiple independent
views of the data is a strong duplicate candidate.  The flagged pairs are
presented with side-by-side EXFO event panels for tech review.

METRICS
-------
  1. Energy profile   — RMS power in 10 chunks (amplitude envelope)
  2. Local variance   — variance in 10 chunks (texture intensity)
  3. Polar area       — trace-as-radius swept area in 10 sectors (shape geometry)
  4. Cumulative sum   — integrated drift sampled at 10 points (accumulated shape)
  5. 2nd-deriv energy — curvature roughness in 10 chunks

USAGE
-----
    python3 rayleighcourse.py /path/to/sor/files/
    python3 rayleighcourse.py /path/to/files/ --route-name "North 288F 3"
    python3 rayleighcourse.py /path/to/files/ --vote-pct 0.01 --min-votes 3
    python3 rayleighcourse.py /path/to/files/ --html report.html --pdf

REQUIREMENTS
------------
    pip install numpy
"""

import os
import sys
import argparse
import json
import subprocess
import webbrowser
import base64
import csv
from datetime import datetime
from itertools import combinations

import numpy as np

# ── Import SOR parsers (co-located in same directory) ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sor_residual_rmse324802 import parse_sor, SORTrace
from rayleigh_fingerprint import (
    extract_fingerprint, RayleighFingerprint, FingerprintMatch,
    collect_sor_files, build_exfo_events,
    generate_pdf_html, _find_chrome, export_pdf,
)

try:
    from sor_reader324741a import parse_sor_full
except ImportError:
    parse_sor_full = None


# ── Shape metrics ───────────────────────────────────────────────────────────

METRIC_NAMES = ['energy', 'lvar', 'polar', 'cumsum', 'd2e', 'growth', 'cos_energy']


def compute_shape_metrics(residual, n_chunks=10):
    """
    Compute 5 independent shape signatures from a Rayleigh residual.

    Each metric captures a different aspect of the trace shape:
      - energy:  RMS power per chunk (amplitude envelope)
      - lvar:    local variance per chunk (texture intensity)
      - polar:   polar-area per chunk (area swept if trace is radius)
      - cumsum:  cumulative sum per chunk (accumulated drift)
      - d2e:     2nd-derivative energy per chunk (curvature roughness)
    """
    r = residual.astype(np.float64)
    n = len(r)
    cs = n // n_chunks

    energy = np.array([np.sqrt(np.mean(r[i*cs:(i+1)*cs]**2))
                        for i in range(n_chunks)])

    lvar = np.array([np.var(r[i*cs:(i+1)*cs])
                      for i in range(n_chunks)])

    r_pos = r - r.min() + 0.001
    dtheta = 2.0 * np.pi / n
    polar = np.array([0.5 * np.sum(r_pos[i*cs:(i+1)*cs]**2) * dtheta
                       for i in range(n_chunks)])

    cumsum = np.cumsum(r)
    cs_sig = np.array([cumsum[min((i+1)*cs - 1, n - 1)] / n
                        for i in range(n_chunks)])

    d2 = np.diff(r, n=2)
    d2cs = len(d2) // n_chunks
    if d2cs < 1:
        d2e = np.zeros(n_chunks)
    else:
        d2e = np.array([np.sqrt(np.mean(d2[i*d2cs:(i+1)*d2cs]**2))
                         for i in range(n_chunks)])

    return {
        'energy': energy,
        'lvar':   lvar,
        'polar':  polar,
        'cumsum': cs_sig,
        'd2e':    d2e,
    }


# ── Multi-metric voting ────────────────────────────────────────────────────

def run_voting(filepaths, vote_pct=0.005, min_votes=5, verbose=False):
    """
    Parse SOR files, extract shape metrics, and vote across all pairs.

    Returns (flagged_pairs, all_results, event_meta) where:
      - flagged_pairs: list of (votes, fiber_a, fiber_b, rank_detail)
      - all_results: sorted list of FingerprintMatch (by RMSE)
      - event_meta: dict of fiber_id -> parse_sor_full result
    """
    # Parse all files
    fingerprints = []
    event_meta = {}
    print(f"\nParsing {len(filepaths)} SOR files...")

    for fp in filepaths:
        _rfp = None
        try:
            sor = parse_sor(fp)
            _rfp = extract_fingerprint(sor)
            fingerprints.append(_rfp)
            if verbose:
                print(f"  {_rfp.fiber_id:<20}  "
                      f"segments={_rfp.num_segments}  "
                      f"rayleigh_pts={_rfp.total_rayleigh_samples:,}")
        except Exception as e:
            print(f"  WARNING: {os.path.basename(fp)}: {e}")

        if parse_sor_full is not None:
            try:
                m = parse_sor_full(fp, trim=True)
                if m:
                    # Use the same fiber_id as the fingerprint so event_meta
                    # keys match the pair comparison results
                    key = _rfp.fiber_id if _rfp else (
                        os.path.basename(fp)
                        .replace('_1550.sor', '')
                        .replace('.sor', '')
                        .replace('.SOR', ''))
                    event_meta[key] = m
            except Exception:
                pass

    if len(fingerprints) < 2:
        print("Need at least 2 valid traces.")
        return [], [], event_meta, np.array([])

    print(f"  {len(fingerprints)} fingerprints extracted "
          f"(dx={fingerprints[0].dx_m:.4f} m)")

    # Compute shape metrics for each fingerprint
    print(f"\nComputing shape metrics (5 metrics × {len(fingerprints)} traces)...")
    fp_metrics = {}
    for fp in fingerprints:
        if fp.segments:
            fp_metrics[fp.fiber_id] = compute_shape_metrics(fp.segments[0].residual)

    fiber_ids = sorted(fp_metrics.keys())

    # Build pair list
    pair_list = []
    for i, a in enumerate(fiber_ids):
        for j in range(i + 1, len(fiber_ids)):
            pair_list.append((a, fiber_ids[j]))

    n_pairs = len(pair_list)
    print(f"  {n_pairs:,} pairs to compare")

    # Also store raw start-aligned residuals for the growth metric
    fp_residuals = {}
    for fp in fingerprints:
        if fp.segments:
            r = fp.segments[0].residual.astype(np.float64)
            r = r - r[0]  # align start to 0
            fp_residuals[fp.fiber_id] = r

    # Rank all pairs on each metric
    print(f"\nRanking pairs across {len(METRIC_NAMES)} metrics...")
    ranks = {mn: {} for mn in METRIC_NAMES}

    # Shape-signature metrics (per-trace, then compare)
    pairwise_metrics = {'growth', 'cos_energy'}
    shape_metrics = [mn for mn in METRIC_NAMES if mn not in pairwise_metrics]
    for mn in shape_metrics:
        dists = []
        for idx, (a, b) in enumerate(pair_list):
            sa = fp_metrics[a][mn]
            sb = fp_metrics[b][mn]
            d = float(np.sqrt(np.mean((sa - sb) ** 2)))
            dists.append((d, idx))
        dists.sort()
        for rank, (d, idx) in enumerate(dists):
            ranks[mn][idx] = rank

    # Growth metric: cumulative absolute difference growth rate (pairwise)
    growth_dists = []
    for idx, (a, b) in enumerate(pair_list):
        if a in fp_residuals and b in fp_residuals:
            ra, rb = fp_residuals[a], fp_residuals[b]
            ml = min(len(ra), len(rb))
            diff = ra[:ml] - rb[:ml]
            cum_abs = np.cumsum(np.abs(diff))
            growth = cum_abs[-1] / np.sqrt(ml)
            growth_dists.append((growth, idx))
        else:
            growth_dists.append((float('inf'), idx))
    growth_dists.sort()
    for rank, (d, idx) in enumerate(growth_dists):
        ranks['growth'][idx] = rank

    # Cosine similarity on energy profile (pairwise — ignores magnitude, compares shape)
    cos_dists = []
    for idx, (a, b) in enumerate(pair_list):
        ea = fp_metrics[a]['energy']
        eb = fp_metrics[b]['energy']
        dot = np.dot(ea, eb)
        na = np.linalg.norm(ea)
        nb = np.linalg.norm(eb)
        cos_sim = dot / (na * nb + 1e-10)
        cos_dists.append((1 - cos_sim, idx))  # lower = more similar
    cos_dists.sort()
    ranks['cos_energy'] = {}
    for rank, (d, idx) in enumerate(cos_dists):
        ranks['cos_energy'][idx] = rank

    # Vote
    threshold = max(1, int(n_pairs * vote_pct))
    all_vote_counts = np.array([
        sum(1 for mn in METRIC_NAMES
            if ranks[mn].get(idx, n_pairs) < threshold)
        for idx in range(n_pairs)])
    flagged = []
    for idx, (a, b) in enumerate(pair_list):
        votes = int(all_vote_counts[idx])
        if votes >= min_votes:
            rank_detail = {mn: ranks[mn].get(idx, -1) for mn in METRIC_NAMES}
            flagged.append((votes, a, b, rank_detail))

    flagged.sort(key=lambda x: (-x[0], x[1]))

    # Post-filter: score each suspect by max splice loss difference
    # across events.  Duplicates have nearly identical splice values;
    # false positives differ by 40+ mdB.
    if flagged and event_meta:
        print(f"\n  Scoring {len(flagged)} suspects by event splice similarity...")

        def _find_meta(fid):
            """Try multiple key formats to find event metadata."""
            if fid in event_meta:
                return event_meta[fid]
            import re
            fid_match = re.search(r'(\d{3,4})$', fid)
            if not fid_match:
                return None
            fid_num = fid_match.group(1).lstrip('0') or '0'
            for key in event_meta:
                key_match = re.search(r'(\d{3,4})$', key)
                if key_match:
                    key_num = key_match.group(1).lstrip('0') or '0'
                    if fid_num == key_num:
                        return event_meta[key]
            return None

        for i, (votes, a, b, rd) in enumerate(flagged):
            ma = _find_meta(a)
            mb = _find_meta(b)
            max_splice_diff = float('inf')
            if ma and mb:
                evts_a = ma.get('events', [])
                evts_b = mb.get('events', [])
                if len(evts_a) == len(evts_b) and evts_a:
                    splice_diffs = [abs(ea['splice_loss'] - eb['splice_loss'])
                                    for ea, eb in zip(evts_a, evts_b)]
                    max_splice_diff = max(splice_diffs) * 1000  # mdB
            rd['max_splice_diff_mdB'] = max_splice_diff

        # Sort: highest votes first, then lowest splice diff
        flagged.sort(key=lambda x: (-x[0], x[3].get('max_splice_diff_mdB', 999)))

    if flagged:
        print(f"\n  Found {len(flagged)} suspected duplicate pair(s) "
              f"(top {vote_pct*100:.1f}% in {min_votes}+ of "
              f"{len(METRIC_NAMES)} metrics):")
        for votes, a, b, rd in flagged:
            detail = ', '.join(f"{mn}:#{rd[mn]+1}" for mn in METRIC_NAMES)
            spl = rd.get('max_splice_diff_mdB', -1)
            spl_str = f", splice_diff={spl:.0f}m" if spl < float('inf') else ""
            conf = ""
            if spl < 25:
                conf = " << HIGH CONFIDENCE"
            elif spl < 50:
                conf = " < review"
            print(f"    {a} ↔ {b}  ({votes}/{len(METRIC_NAMES)} votes{spl_str}{conf})")
    else:
        print(f"\n  No suspects found (threshold: top {vote_pct*100:.1f}%, "
              f"min {min_votes}/{len(METRIC_NAMES)} votes)")

    # Also compute standard RMSE results for reporting
    print(f"\n  Computing Rayleigh RMSE for all pairs...")
    from rayleigh_fingerprint import compare_fingerprints
    all_results = []
    for a_fp, b_fp in combinations(fingerprints, 2):
        try:
            m = compare_fingerprints(a_fp, b_fp)
            all_results.append(m)
        except Exception:
            pass
    all_results.sort(key=lambda r: r.rayleigh_rmse)

    # Tag flagged pairs in results
    flagged_set = {(a, b) for _, a, b, _ in flagged}
    for r in all_results:
        pair = (r.fiber_a, r.fiber_b)
        pair_rev = (r.fiber_b, r.fiber_a)
        if pair in flagged_set or pair_rev in flagged_set:
            r.tier = 2
            r.tier_label = "PROBABLE DUPLICATE (multi-metric)"

    return flagged, all_results, event_meta, all_vote_counts


# ── Reporting ───────────────────────────────────────────────────────────────

def print_report(flagged, all_results, vote_pct, min_votes):
    """Print formatted report."""
    reliable = [r for r in all_results if r.tier_label != "insufficient data"]

    print(f"\n{'═'*76}")
    print(f"  RAYLEIGH COARSE — MULTI-METRIC VOTING REPORT")
    print(f"  Method: 5 shape metrics × vote threshold")
    print(f"  Vote threshold: top {vote_pct*100:.1f}% per metric, "
          f"min {min_votes}/5 votes")
    print(f"{'═'*76}")

    if flagged:
        print(f"\n  ⚑ SUSPECTED DUPLICATES ({len(flagged)} pair(s))")
        print(f"  {'─'*72}")
        for votes, a, b, rd in flagged:
            detail = ', '.join(f"{mn}:#{rd[mn]+1}" for mn in METRIC_NAMES)
            # Find RMSE for this pair
            rmse_str = "?"
            for r in all_results:
                if (r.fiber_a == a and r.fiber_b == b) or \
                   (r.fiber_a == b and r.fiber_b == a):
                    rmse_str = f"{r.rayleigh_rmse*1000:.1f}"
                    break
            print(f"  {a} ↔ {b}  votes={votes}/5  "
                  f"RMSE={rmse_str} mdB")
            print(f"    Ranks: {detail}")
    else:
        print(f"\n  No suspected duplicates found.")

    if reliable:
        all_rmse = [r.rayleigh_rmse for r in reliable
                    if r.rayleigh_rmse < float('inf')]
        print(f"\n  SUMMARY")
        print(f"  {'─'*72}")
        print(f"  Total pairs:     {len(all_results):,}")
        print(f"  Reliable pairs:  {len(reliable):,}")
        print(f"  Mean RMSE:       {np.mean(all_rmse)*1000:.1f} mdB")
        print(f"  Min RMSE:        {np.min(all_rmse)*1000:.1f} mdB")
        print(f"  Suspects:        {len(flagged)}")
    print()


def save_csv(flagged, all_results, path):
    """Save results to CSV."""
    fieldnames = [
        'fiber_a', 'fiber_b',
        'rayleigh_rmse_mdB', 'votes', 'tier_label',
        'rank_energy', 'rank_lvar', 'rank_polar', 'rank_cumsum', 'rank_d2e',
    ]
    # Build lookup for flagged pairs
    flag_lookup = {}
    for votes, a, b, rd in flagged:
        flag_lookup[(a, b)] = (votes, rd)

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            pair = (r.fiber_a, r.fiber_b)
            pair_rev = (r.fiber_b, r.fiber_a)
            fl = flag_lookup.get(pair) or flag_lookup.get(pair_rev)
            votes = fl[0] if fl else 0
            rd = fl[1] if fl else {}
            w.writerow({
                'fiber_a':          r.fiber_a,
                'fiber_b':          r.fiber_b,
                'rayleigh_rmse_mdB': round(r.rayleigh_rmse * 1000, 3),
                'votes':            votes,
                'tier_label':       r.tier_label,
                'rank_energy':      rd.get('energy', ''),
                'rank_lvar':        rd.get('lvar', ''),
                'rank_polar':       rd.get('polar', ''),
                'rank_cumsum':      rd.get('cumsum', ''),
                'rank_d2e':         rd.get('d2e', ''),
            })
    print(f"  Results saved to: {path}")


# ── HTML dashboard ──────────────────────────────────────────────────────────

def generate_html_coarse(flagged, all_results, event_meta,
                          vote_pct, min_votes, route_name=""):
    """Generate HTML dashboard for coarse-resolution results."""
    reliable = [r for r in all_results if r.tier_label != "insufficient data"]
    all_rmse = [r.rayleigh_rmse for r in reliable
                if r.rayleigh_rmse < float('inf')]

    flagged_set = {(a, b) for _, a, b, _ in flagged}
    top10 = reliable[:10]

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    title = (f"Rayleigh Coarse — {route_name}" if route_name
             else "Rayleigh Coarse — Multi-Metric Voting")

    def _pair_obj(r):
        na, nb = r.fiber_a, r.fiber_b
        ma = event_meta.get(na)
        mb = event_meta.get(nb)
        pair = (na, nb)
        pair_rev = (nb, na)
        fl = None
        for v, a, b, rd in flagged:
            if (a, b) == pair or (a, b) == pair_rev:
                fl = (v, rd)
                break
        return {
            'name_a':        na,
            'name_b':        nb,
            'rayleigh_rmse': round(r.rayleigh_rmse * 1000, 3),
            'n_segments':    r.n_matched_segments,
            'n_rayleigh':    r.n_total_rayleigh,
            'events_match':  r.events_match,
            'tier':          r.tier,
            'tier_label':    r.tier_label,
            'votes':         fl[0] if fl else 0,
            'rank_detail':   fl[1] if fl else {},
            'seg_rmses':     [round(s * 1000, 2) for s in r.segment_rmses],
            'events_a':      build_exfo_events(ma),
            'events_b':      build_exfo_events(mb),
            'date_time_a':   ma.get('date_time', 0) if ma else 0,
            'date_time_b':   mb.get('date_time', 0) if mb else 0,
        }

    # Build flagged pair objects
    flagged_objs = []
    for v, a, b, rd in flagged:
        for r in all_results:
            if (r.fiber_a == a and r.fiber_b == b) or \
               (r.fiber_a == b and r.fiber_b == a):
                flagged_objs.append(_pair_obj(r))
                break

    data_json = json.dumps({
        'title':         title,
        'flaggedPairs':  flagged_objs,
        'top10Closest':  [_pair_obj(r) for r in top10],
        'numTraces':     len(event_meta) or '?',
        'numPairs':      len(all_results),
        'numReliable':   len(reliable),
        'votePct':       vote_pct * 100,
        'minVotes':      min_votes,
        'numMetrics':    len(METRIC_NAMES),
        'meanRmse':      round(np.mean(all_rmse) * 1000, 2) if all_rmse else 0,
        'minRmse':       round(np.min(all_rmse) * 1000, 3) if all_rmse else 0,
        'maxRmse':       round(np.max(all_rmse) * 1000, 1) if all_rmse else 0,
        'nFlagged':      len(flagged),
        'generated':     generated,
    })

    # Reuse the same HTML template style from rayleigh_fingerprint
    return COARSE_HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', data_json)


COARSE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rayleigh Coarse — Multi-Metric Voting</title>
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
.suspect-banner{background:#FFF0E0;border:1px solid #E0A050;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#6B4400}
@media(prefers-color-scheme:dark){.suspect-banner{background:#3a2a10;border-color:#886;color:#d4c0a0}}
.suspect-banner strong{font-weight:600}
.near-banner{background:#FFFDE7;border:1px solid #E0D060;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#5D5700}
@media(prefers-color-scheme:dark){.near-banner{background:#2a2a1a;border-color:#665;color:#d4d0a0}}
.near-banner strong{font-weight:600}
.no-dup-banner{background:#E1F5EE;border:1px solid #5DCAA5;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#04342C}
@media(prefers-color-scheme:dark){.no-dup-banner{background:#04342C;border-color:#0F6E56;color:#9FE1CB}}
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
  <span>generated by rayleighcourse.py</span>
</div>
<script>
var DATA = __DATA_PLACEHOLDER__;
var isDark = matchMedia('(prefers-color-scheme:dark)').matches;
var txtSec = isDark ? '#b4b2a9' : '#888780';
var sep = isDark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.1)';
var content = document.getElementById('content');

document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent =
  DATA.numTraces + ' traces \u2022 ' + DATA.numPairs.toLocaleString() + ' pairs \u2022 ' +
  DATA.nFlagged + ' suspects \u2022 generated ' + DATA.generated;
document.getElementById('footer-left').textContent =
  'Multi-metric voting: top ' + DATA.votePct.toFixed(1) + '% in ' + DATA.minVotes + '+/' + DATA.numMetrics + ' metrics';

// Metrics cards
(function() {
  var m = document.getElementById('metrics'); m.className = 'metrics';
  var cards = [
    {label:'pairs analyzed', value:DATA.numReliable.toLocaleString()},
    {label:'mean RMSE', value:DATA.meanRmse.toFixed(1)+' mdB'},
    {label:'min RMSE', value:DATA.minRmse.toFixed(1)+' mdB'},
    {label:'suspects', value:DATA.nFlagged.toString(), sub:'top '+DATA.votePct.toFixed(1)+'% in '+DATA.minVotes+'+/'+DATA.numMetrics},
  ];
  cards.forEach(function(c){
    var d=document.createElement('div');d.className='metric';
    d.innerHTML='<div class="metric-label">'+c.label+'</div><div class="metric-value">'+c.value+'</div>'+(c.sub?'<div class="metric-sub">'+c.sub+'</div>':'');
    m.appendChild(d);
  });
})();

// Event panel builder
function buildEventPanel(pair, bannerClass) {
  var a=pair.name_a, b=pair.name_b;
  var banner=document.createElement('div'); banner.className=bannerClass;
  if (bannerClass==='suspect-banner') {
    banner.innerHTML='<strong>\u26a0 Suspected duplicate:</strong> '+a+' \u2194 '+b+
      ' &mdash; '+pair.votes+'/'+DATA.numMetrics+' metrics vote &mdash; RMSE='+pair.rayleigh_rmse.toFixed(1)+' mdB';
  } else {
    banner.innerHTML='<strong>Closest pair:</strong> '+a+' \u2194 '+b+
      ' &mdash; RMSE='+pair.rayleigh_rmse.toFixed(1)+' mdB ('+pair.n_rayleigh.toLocaleString()+' pts)';
  }
  content.appendChild(banner);

  var sec=document.createElement('div');sec.className='panel';
  sec.innerHTML='<div class="section-label">Key events: '+a+' \u2194 '+b+'</div>';
  var row=document.createElement('div');
  row.style.cssText='display:flex;gap:20px;align-items:flex-start';

  [{nm:a,evts:pair.events_a,color:'#3266ad',ts:pair.date_time_a},
   {nm:b,evts:pair.events_b,color:'#E24B4A',ts:pair.date_time_b}].forEach(function(side){
    var col=document.createElement('div');col.style.cssText='flex:1;min-width:0;overflow-x:auto';
    var tsStr='';
    if(side.ts){var d=new Date(side.ts*1000);tsStr=' <span style="font-weight:400;font-size:11px;color:'+txtSec+'">'+d.toLocaleString()+'</span>';}
    col.innerHTML='<div style="font-size:12px;font-weight:600;margin-bottom:6px;color:'+side.color+'">'+side.nm+tsStr+'</div>';
    if(!side.evts||side.evts.length===0){col.innerHTML+='<div style="font-size:12px;color:#888">No events</div>';}
    else{
      var tbl='<table style="width:100%;border-collapse:collapse;font-size:11px;font-family:\'Courier New\',monospace">';
      tbl+='<tr style="border-bottom:1px solid '+sep+'">'+'<th style="text-align:left;padding:4px 6px;color:'+txtSec+'">#</th>'+'<th style="text-align:left;padding:4px 6px;color:'+txtSec+'">Type</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Dist (km)</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Span (km)</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Splice (dB)</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Refl (dB)</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Atten (dB/km)</th>'+'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Cum Loss</th>'+'</tr>';
      side.evts.forEach(function(e){
        var bg=e.is_end?(isDark?'rgba(255,255,255,.05)':'rgba(0,0,0,.03)'):e.is_refl?(isDark?'rgba(50,102,173,.18)':'rgba(50,102,173,.07)'):'transparent';
        var flag=e.is_end?'\u25a0':e.is_refl?'\u25ba':'\u00a0';
        var reflStr=e.refl!==0?e.refl.toFixed(3):'';
        tbl+='<tr style="background:'+bg+';border-bottom:0.5px solid '+(isDark?'rgba(255,255,255,.04)':'rgba(0,0,0,.04)')+'">'+'<td style="padding:3px 6px">'+e.num+'</td>'+'<td style="padding:3px 6px">'+flag+' '+e.type+'</td>'+'<td style="padding:3px 6px;text-align:right">'+e.dist.toFixed(3)+'</td>'+'<td style="padding:3px 6px;text-align:right">'+e.span.toFixed(3)+'</td>'+'<td style="padding:3px 6px;text-align:right">'+(e.splice>=0?'+':'')+e.splice.toFixed(3)+'</td>'+'<td style="padding:3px 6px;text-align:right">'+reflStr+'</td>'+'<td style="padding:3px 6px;text-align:right">'+e.slope.toFixed(3)+'</td>'+'<td style="padding:3px 6px;text-align:right">'+e.cum.toFixed(3)+'</td>'+'</tr>';
      });
      tbl+='</table>';col.innerHTML+=tbl;
    }
    row.appendChild(col);
  });
  sec.appendChild(row);content.appendChild(sec);
}

// Render suspects
if(DATA.flaggedPairs.length){
  var h=document.createElement('h2');h.textContent='Suspected duplicates (multi-metric voting)';content.appendChild(h);
  DATA.flaggedPairs.forEach(function(d){buildEventPanel(d,'suspect-banner');});
}else{
  var div=document.createElement('div');div.className='no-dup-banner';
  div.textContent='No suspected duplicates found. Closest pair: '+DATA.minRmse.toFixed(1)+' mdB.';content.appendChild(div);
}

// Top 10 closest
if(DATA.top10Closest.length){
  var h2=document.createElement('h2');h2.textContent='Top 10 closest pairs \u2014 event inspection';content.appendChild(h2);
  DATA.top10Closest.forEach(function(d){buildEventPanel(d,'near-banner');});
}
</script>
</body>
</html>"""


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coarse-resolution OTDR duplicate detection via multi-metric voting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'inputs', nargs='+',
        help='SOR files or directories containing SOR files'
    )
    parser.add_argument(
        '--vote-pct', type=float, default=0.005,
        help='Percentile threshold per metric (default: 0.005 = top 0.5%%)'
    )
    parser.add_argument(
        '--min-votes', type=int, default=5,
        help='Minimum metrics a pair must score in to be flagged (default: 5/7)'
    )
    parser.add_argument(
        '--csv', metavar='PATH',
        help='Save results to CSV'
    )
    parser.add_argument(
        '--html', metavar='PATH',
        help='Generate HTML dashboard with event panels'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='Export PDF (requires --html and Chrome)'
    )
    parser.add_argument(
        '--chart-png', metavar='PATH',
        help='Distribution chart PNG to embed at top of PDF'
    )
    parser.add_argument(
        '--route-name', default='',
        help='Route name for report title'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show per-file info'
    )
    parser.add_argument(
        '--open', action='store_true',
        help='Open HTML in browser'
    )
    args = parser.parse_args()

    filepaths = collect_sor_files(args.inputs)
    if not filepaths:
        print("No SOR files found.")
        sys.exit(1)

    print(f"Found {len(filepaths)} SOR file(s)")

    flagged, all_results, event_meta, _vc = run_voting(
        filepaths,
        vote_pct=args.vote_pct,
        min_votes=args.min_votes,
        verbose=args.verbose,
    )

    if not all_results:
        sys.exit(0)

    print_report(flagged, all_results, args.vote_pct, args.min_votes)

    if args.csv:
        save_csv(flagged, all_results, args.csv)

    if args.html:
        print("\nGenerating HTML dashboard...")
        html = generate_html_coarse(
            flagged, all_results, event_meta,
            args.vote_pct, args.min_votes, args.route_name)
        with open(args.html, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Dashboard: {args.html}")

        if args.open:
            webbrowser.open('file://' + os.path.abspath(args.html))

        if args.pdf:
            # Build pairs to show: flagged + top 10 closest
            reliable = [r for r in all_results
                        if r.tier_label != "insufficient data"]
            pairs_to_show = []
            flagged_ids = {(a, b) for _, a, b, _ in flagged}
            # Add flagged first
            for r in all_results:
                pair = (r.fiber_a, r.fiber_b)
                pair_rev = (r.fiber_b, r.fiber_a)
                if pair in flagged_ids or pair_rev in flagged_ids:
                    pairs_to_show.append(r)
            # Then top 10 closest (if not already included)
            shown = {(r.fiber_a, r.fiber_b) for r in pairs_to_show}
            for r in reliable[:10]:
                if (r.fiber_a, r.fiber_b) not in shown:
                    pairs_to_show.append(r)
                    shown.add((r.fiber_a, r.fiber_b))

            if pairs_to_show:
                pdf_html = generate_pdf_html(
                    pairs_to_show, event_meta,
                    0.008, 0.035,
                    len(event_meta) or len(filepaths), len(all_results),
                    args.route_name, args.chart_png)
                pdf_html_path = args.html.replace('.html', '_pdf.html')
                with open(pdf_html_path, 'w', encoding='utf-8') as f:
                    f.write(pdf_html)
                pdf_path = args.html.replace('.html', '.pdf')
                print("Exporting PDF...")
                export_pdf(pdf_html_path, pdf_path)

    return flagged, all_results


if __name__ == '__main__':
    main()
