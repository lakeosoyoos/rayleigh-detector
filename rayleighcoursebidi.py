"""
rayleighcoursebidi.py
=====================
Bidirectional coarse-resolution duplicate detection for OTDR traces.

Takes TWO directories — forward and reverse measurements of the same fibers —
runs multi-metric voting on each direction independently, and unions the
suspect lists.  Duplicates may only be visible from one direction due to
launch-dependent effects, so the union catches more than either alone.

Fibers are matched by number: fiber 034 in the forward folder is assumed
to be the same physical fiber as fiber 034 in the reverse folder.

USAGE
-----
    python3 rayleighcoursebidi.py forward_dir/ reverse_dir/
    python3 rayleighcoursebidi.py forward_dir/ reverse_dir/ --route-name "288F Span 1"
    python3 rayleighcoursebidi.py forward_dir/ reverse_dir/ --html report.html --pdf

REQUIREMENTS
------------
    pip install numpy
"""

import os
import sys
import argparse
import json
import re
import webbrowser
from datetime import datetime
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rayleighcourse import (
    run_voting, compute_shape_metrics, METRIC_NAMES,
    generate_html_coarse, print_report, save_csv,
    COARSE_HTML_TEMPLATE,
)
from rayleigh_fingerprint import (
    collect_sor_files, build_exfo_events,
    generate_pdf_html, export_pdf,
)


# ── Fiber ID normalization ─────────────────────────────────────────────────

def normalize_fiber_id(fiber_id):
    """Extract the numeric fiber number from various naming conventions.
    'TrimmedCHM1CHM20034' -> '034', 'Fiber0034' -> '034', '0034' -> '034'
    """
    # Find the last group of digits
    nums = re.findall(r'\d+', fiber_id)
    if nums:
        return nums[-1].zfill(3)
    return fiber_id


# ── Bidirectional runner ───────────────────────────────────────────────────

def run_bidirectional(fwd_paths, rev_paths,
                       vote_pct=0.005, min_votes=4, verbose=False):
    """
    Run multi-metric voting on forward and reverse directions independently,
    then union the suspect lists.

    Returns (union_flagged, fwd_flagged, rev_flagged,
             fwd_results, rev_results, fwd_meta, rev_meta)
    """
    print("=" * 60)
    print("  FORWARD DIRECTION")
    print("=" * 60)
    fwd_flagged, fwd_results, fwd_meta, _fvc = run_voting(
        fwd_paths, vote_pct=vote_pct, min_votes=min_votes, verbose=verbose)

    print("\n" + "=" * 60)
    print("  REVERSE DIRECTION")
    print("=" * 60)
    rev_flagged, rev_results, rev_meta, _rvc = run_voting(
        rev_paths, vote_pct=vote_pct, min_votes=min_votes, verbose=verbose)

    # Normalize fiber IDs for matching across directions
    def normalize_flagged(flagged):
        """Convert flagged list to use normalized fiber numbers."""
        out = []
        for votes, a, b, rd in flagged:
            na = normalize_fiber_id(a)
            nb = normalize_fiber_id(b)
            pair = tuple(sorted([na, nb]))
            out.append((votes, pair[0], pair[1], rd, a, b))  # keep originals
        return out

    fwd_norm = normalize_flagged(fwd_flagged)
    rev_norm = normalize_flagged(rev_flagged)

    # Build union — track which direction(s) each pair was flagged in
    pair_info = {}  # (norm_a, norm_b) -> {directions, max_votes, details}

    for votes, na, nb, rd, orig_a, orig_b in fwd_norm:
        key = (na, nb)
        if key not in pair_info:
            pair_info[key] = {
                'directions': set(), 'fwd_votes': 0, 'rev_votes': 0,
                'fwd_detail': {}, 'rev_detail': {},
                'fwd_orig': (None, None), 'rev_orig': (None, None),
            }
        pair_info[key]['directions'].add('forward')
        pair_info[key]['fwd_votes'] = votes
        pair_info[key]['fwd_detail'] = rd
        pair_info[key]['fwd_orig'] = (orig_a, orig_b)

    for votes, na, nb, rd, orig_a, orig_b in rev_norm:
        key = (na, nb)
        if key not in pair_info:
            pair_info[key] = {
                'directions': set(), 'fwd_votes': 0, 'rev_votes': 0,
                'fwd_detail': {}, 'rev_detail': {},
                'fwd_orig': (None, None), 'rev_orig': (None, None),
            }
        pair_info[key]['directions'].add('reverse')
        pair_info[key]['rev_votes'] = votes
        pair_info[key]['rev_detail'] = rd
        pair_info[key]['rev_orig'] = (orig_a, orig_b)

    # Sort: both-direction pairs first, then by total votes
    union = []
    for (na, nb), info in pair_info.items():
        n_dirs = len(info['directions'])
        total_votes = info['fwd_votes'] + info['rev_votes']
        union.append((n_dirs, total_votes, na, nb, info))

    union.sort(key=lambda x: (-x[0], -x[1]))

    # Print union report
    print("\n" + "=" * 60)
    print("  BIDIRECTIONAL UNION")
    print("=" * 60)
    print(f"\n  Forward suspects:  {len(fwd_flagged)}")
    print(f"  Reverse suspects:  {len(rev_flagged)}")
    print(f"  Union (unique):    {len(union)}")

    both_dir = [u for u in union if u[0] == 2]
    one_dir = [u for u in union if u[0] == 1]

    if both_dir:
        print(f"\n  ★ FLAGGED IN BOTH DIRECTIONS ({len(both_dir)}):")
        for n_dirs, total_v, na, nb, info in both_dir:
            print(f"    Fiber {na} ↔ {nb}  "
                  f"fwd={info['fwd_votes']}/{len(METRIC_NAMES)} "
                  f"rev={info['rev_votes']}/{len(METRIC_NAMES)} "
                  f"total={total_v}")

    if one_dir:
        print(f"\n  ⚑ FLAGGED IN ONE DIRECTION ({len(one_dir)}):")
        for n_dirs, total_v, na, nb, info in one_dir:
            direction = list(info['directions'])[0]
            votes = info['fwd_votes'] if direction == 'forward' else info['rev_votes']
            print(f"    Fiber {na} ↔ {nb}  "
                  f"{direction}={votes}/{len(METRIC_NAMES)}")

    return union, fwd_flagged, rev_flagged, fwd_results, rev_results, fwd_meta, rev_meta


# ── HTML for bidirectional report ──────────────────────────────────────────

def generate_html_bidir(union, fwd_results, rev_results, fwd_meta, rev_meta,
                         vote_pct, min_votes, route_name=""):
    """Generate HTML dashboard for bidirectional results."""

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    title = (f"Rayleigh Coarse Bidir — {route_name}" if route_name
             else "Rayleigh Coarse — Bidirectional Multi-Metric Voting")

    both_dir = [u for u in union if u[0] == 2]
    one_dir = [u for u in union if u[0] == 1]

    # Build pair objects with events from BOTH directions
    def _pair_obj(na, nb, info):
        # Get events from forward direction
        fwd_a_orig, fwd_b_orig = info['fwd_orig']
        rev_a_orig, rev_b_orig = info['rev_orig']

        fwd_evts_a = build_exfo_events(fwd_meta.get(fwd_a_orig)) if fwd_a_orig else []
        fwd_evts_b = build_exfo_events(fwd_meta.get(fwd_b_orig)) if fwd_b_orig else []
        rev_evts_a = build_exfo_events(rev_meta.get(rev_a_orig)) if rev_a_orig else []
        rev_evts_b = build_exfo_events(rev_meta.get(rev_b_orig)) if rev_b_orig else []

        return {
            'name_a':       f"Fiber {na}",
            'name_b':       f"Fiber {nb}",
            'n_dirs':       len(info['directions']),
            'directions':   list(info['directions']),
            'fwd_votes':    info['fwd_votes'],
            'rev_votes':    info['rev_votes'],
            'total_votes':  info['fwd_votes'] + info['rev_votes'],
            'fwd_events_a': fwd_evts_a,
            'fwd_events_b': fwd_evts_b,
            'rev_events_a': rev_evts_a,
            'rev_events_b': rev_evts_b,
        }

    pair_objs = []
    for n_dirs, total_v, na, nb, info in union:
        pair_objs.append(_pair_obj(na, nb, info))

    data_json = json.dumps({
        'title':         title,
        'pairs':         pair_objs,
        'nBothDir':      len(both_dir),
        'nOneDir':       len(one_dir),
        'nTotal':        len(union),
        'votePct':       vote_pct * 100,
        'minVotes':      min_votes,
        'numMetrics':    len(METRIC_NAMES),
        'generated':     generated,
    })

    return BIDIR_HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', data_json)


BIDIR_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rayleigh Coarse Bidir</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8f7f5;color:#2c2c2a;padding:24px;max-width:1200px;margin:0 auto}
@media(prefers-color-scheme:dark){body{background:#1a1a18;color:#e8e6df}}
h1{font-size:20px;font-weight:500;margin-bottom:4px}
h2{font-size:16px;font-weight:500;margin:24px 0 8px 0}
h3{font-size:13px;font-weight:600;color:#999;letter-spacing:.04em;text-transform:uppercase;margin:12px 0 6px}
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
.both-banner{background:#FAECE7;border:1px solid #F0997B;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#712B13}
@media(prefers-color-scheme:dark){.both-banner{background:#4A1B0C;border-color:#993C1D;color:#F5C4B3}}
.suspect-banner{background:#FFF0E0;border:1px solid #E0A050;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#6B4400}
@media(prefers-color-scheme:dark){.suspect-banner{background:#3a2a10;border-color:#886;color:#d4c0a0}}
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
  <span>generated by rayleighcoursebidi.py</span>
</div>
<script>
var DATA = __DATA_PLACEHOLDER__;
var isDark = matchMedia('(prefers-color-scheme:dark)').matches;
var txtSec = isDark ? '#b4b2a9' : '#888780';
var sep = isDark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.1)';
var content = document.getElementById('content');

document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent =
  DATA.nTotal + ' suspected pairs (' + DATA.nBothDir + ' both dirs, ' +
  DATA.nOneDir + ' one dir) \u2022 generated ' + DATA.generated;
document.getElementById('footer-left').textContent =
  'Bidirectional voting: top ' + DATA.votePct.toFixed(1) + '% in ' +
  DATA.minVotes + '+/' + DATA.numMetrics + ' metrics per direction';

// Metrics
(function(){
  var m=document.getElementById('metrics');m.className='metrics';
  [{label:'total suspects',value:DATA.nTotal.toString()},
   {label:'both directions',value:DATA.nBothDir.toString(),sub:'highest confidence'},
   {label:'one direction',value:DATA.nOneDir.toString(),sub:'review recommended'},
   {label:'vote threshold',value:DATA.minVotes+'/'+DATA.numMetrics,sub:'top '+DATA.votePct.toFixed(1)+'% per metric'}
  ].forEach(function(c){
    var d=document.createElement('div');d.className='metric';
    d.innerHTML='<div class="metric-label">'+c.label+'</div><div class="metric-value">'+c.value+'</div>'+(c.sub?'<div class="metric-sub">'+c.sub+'</div>':'');
    m.appendChild(d);
  });
})();

function buildEvtTable(evts, name, color) {
  if(!evts||evts.length===0) return '<div style="font-size:11px;color:#888">No events</div>';
  var tbl='<table style="width:100%;border-collapse:collapse;font-size:11px;font-family:\'Courier New\',monospace">';
  tbl+='<tr style="border-bottom:1px solid '+sep+'">'
    +'<th style="text-align:left;padding:4px 6px;color:'+txtSec+'">#</th>'
    +'<th style="text-align:left;padding:4px 6px;color:'+txtSec+'">Type</th>'
    +'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Dist (km)</th>'
    +'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Splice (dB)</th>'
    +'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Refl (dB)</th>'
    +'<th style="text-align:right;padding:4px 6px;color:'+txtSec+'">Atten</th></tr>';
  evts.forEach(function(e){
    var bg=e.is_end?(isDark?'rgba(255,255,255,.05)':'rgba(0,0,0,.03)'):e.is_refl?(isDark?'rgba(50,102,173,.18)':'rgba(50,102,173,.07)'):'transparent';
    var reflStr=e.refl!==0?e.refl.toFixed(3):'';
    tbl+='<tr style="background:'+bg+';border-bottom:0.5px solid '+(isDark?'rgba(255,255,255,.04)':'rgba(0,0,0,.04)')+'">'
      +'<td style="padding:3px 6px">'+e.num+'</td>'
      +'<td style="padding:3px 6px">'+e.type+'</td>'
      +'<td style="padding:3px 6px;text-align:right">'+e.dist.toFixed(3)+'</td>'
      +'<td style="padding:3px 6px;text-align:right">'+(e.splice>=0?'+':'')+e.splice.toFixed(3)+'</td>'
      +'<td style="padding:3px 6px;text-align:right">'+reflStr+'</td>'
      +'<td style="padding:3px 6px;text-align:right">'+e.slope.toFixed(3)+'</td></tr>';
  });
  return tbl+'</table>';
}

function buildBidirPanel(pair) {
  var a=pair.name_a, b=pair.name_b;
  var cls = pair.n_dirs===2 ? 'both-banner' : 'suspect-banner';
  var icon = pair.n_dirs===2 ? '\u2605' : '\u26a0';
  var dirText = pair.n_dirs===2
    ? 'Both directions (fwd='+pair.fwd_votes+'/'+DATA.numMetrics+', rev='+pair.rev_votes+'/'+DATA.numMetrics+')'
    : pair.directions.join()+' only ('+pair.total_votes+'/'+DATA.numMetrics+' votes)';

  var banner=document.createElement('div');banner.className=cls;
  banner.innerHTML='<strong>'+icon+' Suspected duplicate:</strong> '+a+' \u2194 '+b+' &mdash; '+dirText;
  content.appendChild(banner);

  var sec=document.createElement('div');sec.className='panel';

  // Forward events
  var fwdHtml = '<h3>Forward direction</h3><div style="display:flex;gap:16px;margin-bottom:12px">';
  fwdHtml += '<div style="flex:1;min-width:0"><div style="font-size:12px;font-weight:600;color:#3266ad;margin-bottom:4px">'+a+'</div>'+buildEvtTable(pair.fwd_events_a)+'</div>';
  fwdHtml += '<div style="flex:1;min-width:0"><div style="font-size:12px;font-weight:600;color:#E24B4A;margin-bottom:4px">'+b+'</div>'+buildEvtTable(pair.fwd_events_b)+'</div>';
  fwdHtml += '</div>';

  // Reverse events
  var revHtml = '<h3>Reverse direction</h3><div style="display:flex;gap:16px">';
  revHtml += '<div style="flex:1;min-width:0"><div style="font-size:12px;font-weight:600;color:#3266ad;margin-bottom:4px">'+a+'</div>'+buildEvtTable(pair.rev_events_a)+'</div>';
  revHtml += '<div style="flex:1;min-width:0"><div style="font-size:12px;font-weight:600;color:#E24B4A;margin-bottom:4px">'+b+'</div>'+buildEvtTable(pair.rev_events_b)+'</div>';
  revHtml += '</div>';

  sec.innerHTML = fwdHtml + revHtml;
  content.appendChild(sec);
}

// Render pairs
if(DATA.pairs.length===0){
  var div=document.createElement('div');div.className='no-dup-banner';
  div.textContent='No suspected duplicates found in either direction.';
  content.appendChild(div);
} else {
  var bothPairs = DATA.pairs.filter(function(p){return p.n_dirs===2;});
  var onePairs = DATA.pairs.filter(function(p){return p.n_dirs===1;});

  if(bothPairs.length){
    var h=document.createElement('h2');
    h.textContent='\u2605 Flagged in both directions ('+bothPairs.length+')';
    content.appendChild(h);
    bothPairs.forEach(function(p){buildBidirPanel(p);});
  }
  if(onePairs.length){
    var h2=document.createElement('h2');
    h2.textContent='\u26a0 Flagged in one direction ('+onePairs.length+')';
    content.appendChild(h2);
    onePairs.forEach(function(p){buildBidirPanel(p);});
  }
}
</script>
</body>
</html>"""


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional coarse-resolution OTDR duplicate detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'forward', help='Forward-direction SOR folder'
    )
    parser.add_argument(
        'reverse', help='Reverse-direction SOR folder'
    )
    parser.add_argument(
        '--vote-pct', type=float, default=0.005,
        help='Percentile threshold per metric (default: 0.005 = top 0.5%%)'
    )
    parser.add_argument(
        '--min-votes', type=int, default=5,
        help='Minimum metrics to be flagged (default: 5/7)'
    )
    parser.add_argument(
        '--html', metavar='PATH',
        help='Generate HTML dashboard'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='Export PDF (requires --html and Chrome)'
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

    fwd_paths = collect_sor_files([args.forward])
    rev_paths = collect_sor_files([args.reverse])

    if not fwd_paths or not rev_paths:
        print("Need SOR files in both directories.")
        sys.exit(1)

    print(f"Forward: {len(fwd_paths)} files from {args.forward}")
    print(f"Reverse: {len(rev_paths)} files from {args.reverse}")

    (union, fwd_flagged, rev_flagged,
     fwd_results, rev_results,
     fwd_meta, rev_meta) = run_bidirectional(
        fwd_paths, rev_paths,
        vote_pct=args.vote_pct,
        min_votes=args.min_votes,
        verbose=args.verbose,
    )

    if args.html:
        print("\nGenerating bidirectional HTML dashboard...")
        html = generate_html_bidir(
            union, fwd_results, rev_results, fwd_meta, rev_meta,
            args.vote_pct, args.min_votes, args.route_name)
        with open(args.html, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Dashboard: {args.html}")

        if args.open:
            webbrowser.open('file://' + os.path.abspath(args.html))

        if args.pdf:
            # Build a simple PDF from the union pairs
            # Collect FingerprintMatch results for flagged pairs
            all_meta = {**fwd_meta, **rev_meta}
            # Use forward results to find RMSE for flagged pairs
            all_rmse_results = fwd_results + rev_results
            flagged_norm = set()
            for n_dirs, total_v, na, nb, info in union:
                flagged_norm.add((na, nb))

            pairs_for_pdf = []
            seen = set()
            for r in all_rmse_results:
                ra = normalize_fiber_id(r.fiber_a)
                rb = normalize_fiber_id(r.fiber_b)
                pair = tuple(sorted([ra, rb]))
                if pair in flagged_norm and pair not in seen:
                    pairs_for_pdf.append(r)
                    seen.add(pair)

            if pairs_for_pdf:
                n_traces = max(len(fwd_paths), len(rev_paths))
                n_pairs = len(fwd_results) + len(rev_results)
                pdf_html = generate_pdf_html(
                    pairs_for_pdf, all_meta,
                    0.008, 0.035, n_traces, n_pairs,
                    args.route_name)
                pdf_html_path = args.html.replace('.html', '_pdf.html')
                with open(pdf_html_path, 'w', encoding='utf-8') as f:
                    f.write(pdf_html)
                pdf_path = args.html.replace('.html', '.pdf')
                print("Exporting PDF...")
                export_pdf(pdf_html_path, pdf_path)


if __name__ == '__main__':
    main()
