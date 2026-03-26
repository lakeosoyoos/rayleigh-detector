"""
rayleighcourse_dashboard.py
===========================
Generates an HTML + PDF dashboard for the Rayleigh Coarse multi-metric
voting duplicate detection system.

Dashboard layout:
  1. Vote distribution histogram (how many pairs got 0,1,2,...,7 votes)
  2. Vote breakdown spreadsheet (which metrics voted for each suspect)
  3. Confidence-ranked suspect list (sorted by splice similarity)
  4. Side-by-side EXFO event panels for each suspect

Usage:
    python3 rayleighcourse_dashboard.py /path/to/sor/folder/
    python3 rayleighcourse_dashboard.py /path/to/folder/ --route-name "North 288F 3"
    python3 rayleighcourse_dashboard.py /path/to/folder/ --open
"""

import os
import sys
import re
import argparse
import json
import base64
import subprocess
import webbrowser
from datetime import datetime
from io import BytesIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/Desktop'))
from rayleighcourse import (
    run_voting, METRIC_NAMES, compute_shape_metrics,
)
from rayleigh_fingerprint import (
    collect_sor_files, build_exfo_events, _find_chrome,
)


# ── Metric labels ───────────────────────────────────────────────────────────

METRIC_LABELS = {
    'energy':     'Energy',
    'lvar':       'Variance',
    'polar':      'Polar Area',
    'cumsum':     'Cum. Sum',
    'd2e':        '2nd Deriv',
    'growth':     'Growth',
    'cos_energy': 'Cosine Sim',
}


# ── Chart generation ────────────────────────────────────────────────────────

def generate_vote_chart(vote_counts, n_suspects):
    """Generate vote distribution histogram as base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))
    bins = np.arange(-0.5, 8.5, 1)
    counts_hist, edges, patches = ax.hist(
        vote_counts, bins=bins, color='#4A90D9',
        edgecolor='white', linewidth=1.5, alpha=0.85)

    cmap = {0: '#4A90D9', 1: '#4A90D9', 2: '#6BAED6', 3: '#E0C060',
            4: '#E0A050', 5: '#E34B4A', 6: '#C0392B', 7: '#8B0000'}
    for patch, le in zip(patches, edges[:-1]):
        v = int(le + 0.5)
        patch.set_facecolor(cmap.get(v, '#4A90D9'))
        if v >= 4:
            patch.set_edgecolor('darkred')
            patch.set_linewidth(2)

    for i, (c, e) in enumerate(zip(counts_hist, edges[:-1])):
        if c > 0:
            ax.text(int(e + 0.5), c + max(counts_hist) * 0.01,
                    f'{int(c):,}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.axvspan(4.5, 7.5, alpha=0.06, color='red')
    if n_suspects > 0 and counts_hist[5] > 0:
        ax.annotate(f'{n_suspects} suspects\n(5+ votes)',
                    xy=(5, counts_hist[5]),
                    xytext=(6.2, max(counts_hist) * 0.5),
                    fontsize=11, fontweight='bold', color='#E34B4A',
                    arrowprops=dict(arrowstyle='->', color='#E34B4A', lw=2))

    n_total = len(vote_counts)
    stats = (f'Total pairs: {n_total:,}\n'
             f'0 votes: {int(counts_hist[0]):,} ({counts_hist[0]/n_total*100:.1f}%)\n'
             f'1 vote: {int(counts_hist[1]):,}\n'
             f'5+ votes: {n_suspects}')
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#ccc', alpha=0.9))

    ax.set_xlabel('Votes out of 7 metrics', fontsize=12)
    ax.set_ylabel('Number of pairs', fontsize=12)
    ax.set_title(f'Multi-Metric Vote Distribution — {n_total:,} pairs, '
                 f'7 metrics, top 0.5% threshold',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_xlim(-0.7, 7.7)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    return b64


# ── Dashboard HTML builder ──────────────────────────────────────────────────

def _find_event_meta(fid, event_meta):
    """Try multiple key formats to find event metadata."""
    if fid in event_meta:
        return event_meta[fid]
    # Extract trailing digits from fiber ID (e.g. 'Fiber0034' -> '034' or '0034')
    fid_match = re.search(r'(\d{3,4})$', fid)
    if not fid_match:
        return None
    fid_num = fid_match.group(1).lstrip('0') or '0'
    for key in event_meta:
        # Extract trailing digits from meta key (e.g. 'TrimmedCHM3CHM40034' -> '034')
        key_match = re.search(r'(\d{3,4})$', key)
        if key_match:
            key_num = key_match.group(1).lstrip('0') or '0'
            if fid_num == key_num:
                return event_meta[key]
    return None


def build_dashboard(flagged, all_results, event_meta, vote_counts,
                     vote_pct, min_votes, route_name=""):
    """Build complete HTML dashboard string."""

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    n_total = len(all_results)
    title = (f"Rayleigh Coarse — {route_name}" if route_name
             else "Rayleigh Coarse — Multi-Metric Voting")

    # Chart
    chart_b64 = generate_vote_chart(vote_counts, len(flagged))

    # Build vote breakdown table
    vote_header = ''.join(
        f'<th>{METRIC_LABELS.get(mn, mn)}</th>' for mn in METRIC_NAMES)

    vote_rows = ''
    for votes, a, b, rd in flagged:
        spl = rd.get('max_splice_diff_mdB', float('inf'))
        conf = rd.get('confidence', '')
        if conf == 'HIGH':
            conf_cls = 'conf-high'
            conf_label = 'HIGH'
        elif conf == 'REVIEW':
            conf_cls = 'conf-review'
            conf_label = 'REVIEW'
        else:
            conf_cls = 'conf-low'
            conf_label = ''

        spl_str = f'{spl:.0f}' if spl < float('inf') else '?'
        cells = (f'<td class="pair-cell">{a} \u2194 {b}</td>'
                 f'<td class="center bold">{votes}/7</td>'
                 f'<td class="center {conf_cls}">{spl_str}</td>'
                 f'<td class="center {conf_cls}">{conf_label}</td>')
        for mn in METRIC_NAMES:
            r = rd.get(mn, -1)
            if isinstance(r, dict):
                r = r.get('rank', -1)
            voted = r < max(1, int(n_total * vote_pct))
            if voted:
                cells += f'<td class="center voted">#{r+1}</td>'
            else:
                cells += f'<td class="center novote">#{r+1}</td>'

        row_cls = f' class="{conf_cls}-row"' if conf_cls == 'conf-high' else ''
        vote_rows += f'<tr{row_cls}>{cells}</tr>\n'

    # Build event panels
    event_panels = ''
    for i, (votes, a, b, rd) in enumerate(flagged):
        spl = rd.get('max_splice_diff_mdB', float('inf'))
        conf = rd.get('confidence', '')
        if conf == 'HIGH':
            banner_cls = 'banner-high'
            icon = '\u2605 HIGH CONFIDENCE:'
        elif conf == 'REVIEW':
            banner_cls = 'banner-review'
            icon = '\u26a0 Review:'
        else:
            banner_cls = 'banner-low'
            icon = 'Suspect:'

        spl_str = f'splice diff = {spl:.0f} mdB' if spl < float('inf') else ''

        ma = _find_event_meta(a, event_meta)
        mb = _find_event_meta(b, event_meta)
        evts_a = build_exfo_events(ma) if ma else []
        evts_b = build_exfo_events(mb) if mb else []

        evt_a_html = _evt_table_html(a, evts_a, '#3266ad')
        evt_b_html = _evt_table_html(b, evts_b, '#E24B4A')

        event_panels += f'''
        <div class="pair-block">
          <div class="{banner_cls}">
            <strong>{icon}</strong> {a} \u2194 {b} &mdash; {votes}/7 votes
            &mdash; {spl_str}
          </div>
          <div class="evt-row">
            <div class="evt-col">{evt_a_html}</div>
            <div class="evt-col">{evt_b_html}</div>
          </div>
        </div>'''

    return DASHBOARD_HTML.format(
        title=title,
        subtitle=f'{route_name} &bull; {n_total:,} pairs &bull; '
                 f'{len(flagged)} suspects &bull; generated {generated}',
        chart_b64=chart_b64,
        vote_header=vote_header,
        vote_rows=vote_rows,
        n_suspects=len(flagged),
        n_high=sum(1 for _, _, _, rd in flagged
                   if rd.get('confidence') == 'HIGH'),
        n_review=sum(1 for _, _, _, rd in flagged
                     if rd.get('confidence') == 'REVIEW'),
        event_panels=event_panels,
        footer_left=f'{route_name} &bull; {len(flagged)} suspects &bull; '
                    f'7 metrics &bull; top {vote_pct*100:.1f}%',
        generated=generated,
    )


def _evt_table_html(name, evts, color):
    """Build event table HTML for one fiber."""
    html = (f'<div style="font-size:11px;font-weight:600;color:{color};'
            f'margin-bottom:4px">{name}</div>')
    if not evts:
        return html + '<div style="color:#888;font-size:10px">No events</div>'

    html += ('<table class="evt"><tr>'
             '<th>#</th><th>Type</th>'
             '<th class="r">Dist (km)</th>'
             '<th class="r">Splice (dB)</th>'
             '<th class="r">Refl (dB)</th>'
             '<th class="r">Atten</th></tr>')
    for e in evts:
        refl = f'{e["refl"]:.3f}' if e['refl'] != 0 else ''
        bg = ('#f0f4fa' if e['is_refl'] and not e['is_end']
              else '#f5f5f3' if e['is_end'] else '')
        st = f' style="background:{bg}"' if bg else ''
        sign = '+' if e['splice'] >= 0 else ''
        html += (f'<tr{st}>'
                 f'<td>{e["num"]}</td>'
                 f'<td>{e["type"]}</td>'
                 f'<td class="r">{e["dist"]:.4f}</td>'
                 f'<td class="r">{sign}{e["splice"]:.3f}</td>'
                 f'<td class="r">{refl}</td>'
                 f'<td class="r">{e["slope"]:.3f}</td>'
                 f'</tr>')
    return html + '</table>'


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{ size: landscape; margin: 10mm; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        color:#2c2c2a; padding:16px; font-size:11px; max-width:1200px; margin:0 auto; }}
h1 {{ font-size:18px; font-weight:500; margin-bottom:2px; }}
h2 {{ font-size:14px; font-weight:500; margin:20px 0 8px; }}
.subtitle {{ font-size:11px; color:#888; margin-bottom:14px; }}
.chart-img {{ width:100%; border-radius:8px; border:1px solid #ddd; margin-bottom:16px; }}

/* Summary cards */
.summary {{ display:flex; gap:12px; margin-bottom:16px; }}
.card {{ flex:1; background:#fff; border:1px solid rgba(0,0,0,.08); border-radius:10px; padding:12px 14px; }}
.card-label {{ font-size:10px; color:#999; margin-bottom:2px; }}
.card-value {{ font-size:22px; font-weight:600; }}
.card-sub {{ font-size:10px; color:#999; }}
.card-high {{ border-color:#F0997B; background:#FEF6F3; }}
.card-high .card-value {{ color:#C0392B; }}

/* Vote table */
.vote-table {{ width:100%; border-collapse:collapse; font-size:9px;
               font-family:'SF Mono','Courier New',monospace; margin-bottom:4px; }}
.vote-table th {{ background:#f4f3f0; padding:5px 6px; text-align:center;
                  font-weight:600; border:0.5px solid #ddd; font-size:8px; color:#555; }}
.vote-table td {{ padding:4px 6px; border:0.5px solid #ddd; }}
.pair-cell {{ text-align:left !important; font-weight:600; }}
.center {{ text-align:center; }}
.bold {{ font-weight:600; }}
.voted {{ background:#E8F5E9; font-weight:600; color:#2E7D32; }}
.novote {{ color:#ccc; }}
.conf-high {{ color:#C0392B; font-weight:700; }}
.conf-review {{ color:#E67E22; font-weight:600; }}
.conf-low {{ color:#999; }}
.conf-high-row {{ background:#FEF6F3; }}
.vote-note {{ font-size:8px; color:#999; margin-bottom:16px; }}

/* Event panels */
.pair-block {{ margin-bottom:14px; }}
.banner-high {{ background:#FAECE7; border:1px solid #F0997B; border-radius:8px;
                padding:10px 14px; margin-bottom:8px; font-size:12px; color:#712B13; }}
.banner-review {{ background:#FFF8E1; border:1px solid #E0C060; border-radius:8px;
                  padding:10px 14px; margin-bottom:8px; font-size:12px; color:#5D5200; }}
.banner-low {{ background:#F5F5F3; border:1px solid #ddd; border-radius:8px;
               padding:10px 14px; margin-bottom:8px; font-size:12px; color:#666; }}
.banner-high strong, .banner-review strong {{ font-weight:700; }}
.evt-row {{ display:flex; gap:14px; }}
.evt-col {{ flex:1; min-width:0; overflow:hidden; }}
.evt {{ width:100%; border-collapse:collapse; font-size:9px;
        font-family:'Courier New',monospace; }}
.evt th {{ background:#f4f3f0; padding:3px 5px; text-align:left;
           font-weight:600; border-bottom:1px solid #ccc; font-size:8px; color:#666; }}
.evt td {{ padding:2px 5px; border-bottom:0.5px solid #eee; }}
.evt .r {{ text-align:right; }}

.footer {{ margin-top:16px; padding-top:8px; border-top:0.5px solid #ddd;
           font-size:9px; color:#999; display:flex; justify-content:space-between; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>

<div class="summary">
  <div class="card card-high">
    <div class="card-label">high confidence</div>
    <div class="card-value">{n_high}</div>
    <div class="card-sub">splice diff &lt; 25 mdB</div>
  </div>
  <div class="card">
    <div class="card-label">review</div>
    <div class="card-value">{n_review}</div>
    <div class="card-sub">splice diff 25-50 mdB</div>
  </div>
  <div class="card">
    <div class="card-label">total suspects</div>
    <div class="card-value">{n_suspects}</div>
    <div class="card-sub">5+ of 7 metrics vote</div>
  </div>
</div>

<img src="data:image/png;base64,{chart_b64}" class="chart-img" />

<h2>Vote Breakdown &mdash; Metric Rankings for Each Suspect</h2>
<table class="vote-table">
<tr><th style="text-align:left">Pair</th><th>Votes</th>
    <th>Splice<br/>Diff (m)</th><th>Conf.</th>{vote_header}</tr>
{vote_rows}
</table>
<div class="vote-note">
  Green cells = metric voted for this pair (ranked in top 0.5%).
  Splice Diff = max difference in splice loss across all events (lower = more likely duplicate).
  HIGH = &lt;25 mdB, REVIEW = 25-50 mdB.
</div>

<h2>Key Events &mdash; Side-by-Side Comparison</h2>
{event_panels}

<div class="footer">
  <span>{footer_left}</span>
  <span>generated by rayleighcourse_dashboard.py</span>
</div>

</body>
</html>'''


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Rayleigh Coarse dashboard with vote distribution, "
                    "metric breakdown, and event panels.")
    parser.add_argument('inputs', nargs='+',
                        help='SOR files or directories')
    parser.add_argument('--route-name', default='',
                        help='Route name for title')
    parser.add_argument('--vote-pct', type=float, default=0.005)
    parser.add_argument('--min-votes', type=int, default=5)
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML path')
    parser.add_argument('--pdf', action='store_true',
                        help='Also export PDF via Chrome')
    parser.add_argument('--open', action='store_true',
                        help='Open in browser')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    filepaths = collect_sor_files(args.inputs)
    if not filepaths:
        print("No SOR files found.")
        sys.exit(1)

    print(f"Found {len(filepaths)} SOR files")

    # Run voting with the specified min_votes threshold
    flagged, all_results, event_meta, _vc = run_voting(
        filepaths,
        vote_pct=args.vote_pct,
        min_votes=args.min_votes,
        verbose=args.verbose)

    # Post-filter: compute splice diff for each suspect, then gap-analyze
    if flagged:
        for i, (votes, a, b, rd) in enumerate(flagged):
            ma = _find_event_meta(a, event_meta)
            mb = _find_event_meta(b, event_meta)
            max_splice_diff = float('inf')
            if ma and mb:
                evts_a = ma.get('events', [])
                evts_b = mb.get('events', [])
                if len(evts_a) == len(evts_b) and evts_a:
                    splice_diffs = [abs(ea['splice_loss'] - eb['splice_loss'])
                                    for ea, eb in zip(evts_a, evts_b)]
                    max_splice_diff = max(splice_diffs) * 1000
            rd['max_splice_diff_mdB'] = max_splice_diff

        # Sort by splice diff for gap analysis
        flagged.sort(key=lambda x: x[3].get('max_splice_diff_mdB', 999))

        # Gap analysis: find natural break in splice diffs
        splice_vals = [c[3].get('max_splice_diff_mdB', 999)
                       for c in flagged if c[3].get('max_splice_diff_mdB', 999) < float('inf')]
        gap_idx = -1
        best_gap = 1.0
        for i in range(1, len(splice_vals)):
            if splice_vals[i-1] > 0:
                ratio = splice_vals[i] / splice_vals[i-1]
                if ratio > best_gap:
                    best_gap = ratio
                    gap_idx = i

        if best_gap >= 1.5 and gap_idx > 0:
            splice_threshold = splice_vals[gap_idx - 1]
            print(f"\n  Splice gap analysis: natural break at "
                  f"{splice_vals[gap_idx-1]:.0f} mdB -> "
                  f"{splice_vals[gap_idx]:.0f} mdB ({best_gap:.1f}x gap)")
            print(f"  {gap_idx} pair(s) below gap = HIGH CONFIDENCE duplicates")
            # Tag pairs below the gap
            for votes, a, b, rd in flagged:
                spl = rd.get('max_splice_diff_mdB', 999)
                if spl <= splice_threshold:
                    rd['confidence'] = 'HIGH'
                elif spl < splice_threshold * 2:
                    rd['confidence'] = 'REVIEW'
                else:
                    rd['confidence'] = ''
        else:
            # No gap found — use fixed thresholds
            for votes, a, b, rd in flagged:
                spl = rd.get('max_splice_diff_mdB', 999)
                if spl < 25:
                    rd['confidence'] = 'HIGH'
                elif spl < 50:
                    rd['confidence'] = 'REVIEW'
                else:
                    rd['confidence'] = ''

    if not all_results:
        sys.exit(0)

    # Compute vote counts for histogram
    from rayleigh_fingerprint import extract_fingerprint
    from sor_residual_rmse324802 import parse_sor as _parse_sor
    from itertools import combinations

    fingerprints = []
    for fp in filepaths:
        try:
            sor = _parse_sor(fp)
            rfp = extract_fingerprint(sor)
            fingerprints.append(rfp)
        except:
            pass

    profiles = {}
    fp_res = {}
    for fp in fingerprints:
        if fp.segments:
            profiles[fp.fiber_id] = compute_shape_metrics(fp.segments[0].residual)
            r = fp.segments[0].residual.astype(np.float64)
            fp_res[fp.fiber_id] = r - r[0]

    fiber_ids = sorted(profiles.keys())
    pair_list = [(fiber_ids[i], fiber_ids[j])
                 for i in range(len(fiber_ids))
                 for j in range(i + 1, len(fiber_ids))]
    n_pairs = len(pair_list)

    # Rank all metrics
    pairwise_mns = {'growth', 'cos_energy'}
    shape_mns = [mn for mn in METRIC_NAMES if mn not in pairwise_mns]
    ranks = {}
    for mn in shape_mns:
        dists = [(float(np.sqrt(np.mean(
            (profiles[a][mn] - profiles[b][mn]) ** 2))), idx)
            for idx, (a, b) in enumerate(pair_list)]
        dists.sort()
        ranks[mn] = {idx: rank for rank, (d, idx) in enumerate(dists)}

    # Growth
    dists = []
    for idx, (a, b) in enumerate(pair_list):
        if a in fp_res and b in fp_res:
            ra, rb = fp_res[a], fp_res[b]
            ml = min(len(ra), len(rb))
            cum = np.cumsum(np.abs(ra[:ml] - rb[:ml]))
            dists.append((cum[-1] / np.sqrt(ml), idx))
        else:
            dists.append((float('inf'), idx))
    dists.sort()
    ranks['growth'] = {idx: rank for rank, (d, idx) in enumerate(dists)}

    # Cosine
    dists = []
    for idx, (a, b) in enumerate(pair_list):
        ea, eb = profiles[a]['energy'], profiles[b]['energy']
        cs = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-10)
        dists.append((1 - cs, idx))
    dists.sort()
    ranks['cos_energy'] = {idx: rank for rank, (d, idx) in enumerate(dists)}

    threshold = max(1, int(n_pairs * args.vote_pct))
    vote_counts = np.array([
        sum(1 for mn in METRIC_NAMES
            if ranks[mn].get(idx, n_pairs) < threshold)
        for idx in range(n_pairs)])

    # Generate dashboard
    route = args.route_name or os.path.basename(args.inputs[0].rstrip('/'))
    html = build_dashboard(flagged, all_results, event_meta,
                            vote_counts, args.vote_pct, args.min_votes,
                            route)

    out_path = args.output or os.path.join(
        os.path.expanduser('~/Desktop'),
        f'rayleighcourse_{route.lower().replace(" ", "_")}_dashboard.html')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nDashboard: {out_path}")
    print(f"  Size: {len(html):,} bytes")

    if args.pdf:
        pdf_path = out_path.replace('.html', '.pdf')
        chrome = _find_chrome()
        if chrome:
            print("Exporting PDF...")
            result = subprocess.run(
                [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
                 '--run-all-compositor-stages-before-draw',
                 '--virtual-time-budget=5000',
                 f'--print-to-pdf={os.path.abspath(pdf_path)}',
                 '--print-to-pdf-no-header', '--no-pdf-header-footer',
                 'file://' + os.path.abspath(out_path)],
                capture_output=True, timeout=120)
            if result.returncode == 0:
                print(f"  PDF: {pdf_path}")
            else:
                print(f"  PDF failed: {result.stderr.decode()[:200]}")

    if args.open:
        webbrowser.open('file://' + os.path.abspath(out_path))


if __name__ == '__main__':
    main()
