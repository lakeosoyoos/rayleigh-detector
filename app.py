"""
Rayleigh Duplicate Detector — Streamlit App
============================================
Web-based OTDR trace duplicate detection using Rayleigh backscatter fingerprinting.

Launch:  streamlit run app.py
"""

import os
import sys
import tempfile
import io
import subprocess
from contextlib import redirect_stdout

import streamlit as st
import numpy as np

# Ensure our modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sor_residual_rmse324802 import parse_sor
from rayleigh_fingerprint import (
    run_batch, collect_sor_files, generate_html,
    generate_pdf_html, export_pdf, _find_chrome,
    EXACT_DUP_THRESHOLD, PROBABLE_DUP_THRESHOLD,
    DEFAULT_EVENT_BUFFER, MIN_RAYLEIGH_POINTS, MIN_RAYLEIGH_PCT,
)
from rayleighcourse import run_voting, generate_html_coarse, METRIC_NAMES, compute_shape_metrics
from rayleighcourse_dashboard import build_dashboard, _find_event_meta, generate_vote_chart


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rayleigh Duplicate Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

/* ── Reset & base ────────────────────────────────────────────── */
#MainMenu       { visibility: hidden; }
footer          { visibility: hidden; }
header          { visibility: hidden; }
.stDeployButton { display: none; }

html, body, [class*="css"], .stMarkdown, p, label, span, div {
    font-family: 'Nunito', 'Segoe UI', Arial, sans-serif !important;
}
.main .block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Top utility strip ───────────────────────────────────────── */
.tc-topbar {
    background: #1C2526;
    color: #cccccc;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Nunito', sans-serif;
    padding: 7px 36px;
    display: flex;
    justify-content: flex-end;
    gap: 28px;
    letter-spacing: 0.4px;
}
.tc-topbar span { color: #aaaaaa; cursor: default; }

/* ── Main nav bar ────────────────────────────────────────────── */
.tc-navbar {
    background: #ffffff;
    border-bottom: 2px solid #eeeeee;
    padding: 16px 36px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.tc-logo-icon {
    font-size: 30px;
    font-weight: 900;
    color: #E8461E;
    line-height: 1;
    font-family: 'Nunito', sans-serif;
    transform: skewX(-8deg);
    display: inline-block;
}
.tc-logo-name {
    font-size: 22px;
    font-weight: 900;
    color: #1a1a1a !important;
    font-family: 'Nunito', sans-serif;
    letter-spacing: -0.3px;
}
.tc-navbar-spacer { flex: 1; }
.tc-contact-btn {
    border: 2px solid #E8461E;
    color: #1a1a1a;
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 13px;
    padding: 8px 18px;
    position: relative;
    cursor: default;
    letter-spacing: 0.2px;
}
.tc-contact-btn::after {
    content: "";
    position: absolute;
    bottom: -2px;
    right: -2px;
    width: 10px;
    height: 10px;
    background: #E8461E;
}

/* ── Hero banner ─────────────────────────────────────────────── */
.tc-hero {
    background: linear-gradient(105deg, #E8461E 55%, #c23610 100%);
    padding: 42px 36px 38px 36px;
    margin-bottom: 32px;
}
.tc-hero h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 34px;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 10px 0;
    line-height: 1.15;
    letter-spacing: -0.3px;
}
.tc-hero p {
    font-family: 'Nunito', sans-serif;
    font-size: 15px;
    color: rgba(255,255,255,0.88);
    margin: 0;
    font-weight: 600;
}

/* ── Cards ───────────────────────────────────────────────────── */
.tc-card {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-top: 4px solid #E8461E;
    border-radius: 4px;
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.tc-card-title {
    font-family: 'Nunito', sans-serif;
    font-size: 16px;
    font-weight: 900;
    color: #1a1a1a;
    margin: 0 0 14px 0;
    letter-spacing: -0.1px;
}

/* ── Checklist ───────────────────────────────────────────────── */
.tc-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.tc-list li {
    font-family: 'Nunito', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #333;
    padding: 5px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    line-height: 1.5;
}
.tc-list li::before {
    content: "▸";
    color: #E8461E;
    font-weight: 900;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Stat tiles ──────────────────────────────────────────────── */
.tc-stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 0 0 28px 0;
}
.tc-stat {
    background: #fff;
    border: 1px solid #e5e5e5;
    border-top: 4px solid #E8461E;
    border-radius: 4px;
    padding: 16px 20px;
    min-width: 140px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.tc-stat-label {
    font-family: 'Nunito', sans-serif;
    font-size: 11px;
    font-weight: 800;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.tc-stat-value {
    font-family: 'Nunito', sans-serif;
    font-size: 30px;
    font-weight: 900;
    color: #1a1a1a;
    line-height: 1;
}
.tc-stat-sub {
    font-family: 'Nunito', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #777;
    margin-top: 3px;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1C2526 !important;
    border-right: 3px solid #E8461E !important;
    width: 620px !important;
    min-width: 620px !important;
    max-width: 620px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem;
    width: 620px !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stSidebar"] button[title="Collapse sidebar"],
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] {
    display: none !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    font-family: 'Nunito', sans-serif !important;
    color: #e8e8e8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    text-align: center !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background: #243030 !important;
    border-color: #E8461E !important;
    color: #e8e8e8 !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2e3d3d !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #aaa !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background-color: #E8461E !important;
    border-color: #E8461E !important;
    color: white !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 3px !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background-color: #c23610 !important;
    border-color: #c23610 !important;
}
.stButton > button,
.stDownloadButton > button {
    border-color: #E8461E !important;
    color: #E8461E !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 3px !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #c23610 !important;
    color: #c23610 !important;
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background-color: #E8461E !important;
}

/* ── Radio ───────────────────────────────────────────────────── */
.stRadio [role="radiogroup"] label {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    background-color: transparent !important;
    border-color: transparent !important;
}
.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label:has(input:checked) {
    background-color: transparent !important;
    border-color: transparent !important;
    color: inherit !important;
}
:root { --primary-color: #E8461E !important; }
[data-baseweb="radio"] [role="radio"][aria-checked="true"] div {
    background-color: #E8461E !important;
    border-color: #E8461E !important;
}
[data-baseweb="radio"] [role="radio"] div {
    border-color: #E8461E !important;
}

/* ── Links ───────────────────────────────────────────────────── */
a { color: #E8461E !important; }

/* ── Expander ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e0e0e0 !important;
    border-radius: 4px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: #1a1a1a !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary:hover { color: #E8461E !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary * {
    color: #1a1a1a !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tc-topbar">
    <span>OTDR QC Tools</span>
    <span>Help</span>
</div>
<div class="tc-navbar">
    <div class="tc-logo-icon">↗</div>
    <a href="/" target="_self" class="tc-logo-name" style="text-decoration:none; cursor:pointer;">Rayleigh Duplicate Detector</a>
    <div class="tc-navbar-spacer"></div>
    <div class="tc-contact-btn">OTDR QC &nbsp; ▸</div>
</div>
""", unsafe_allow_html=True)


# ── Password protection ──────────────────────────────────────────────────────

def check_password():
    """Return True if user entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    try:
        correct = st.secrets["passwords"]["app_password"]
    except (KeyError, FileNotFoundError):
        return True  # No password configured — allow access
    pwd = st.text_input("Enter password", type="password", key="pwd_input")
    if pwd:
        if pwd == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()


# ── Session state init ───────────────────────────────────────────────────────

for key in ["pdf_bytes", "pdf_name", "html_report", "html_name",
            "summary", "log_output", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "log_output" not in st.session_state:
    st.session_state.log_output = ""
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload SOR Files")

    mode = st.radio(
        "Analysis Mode",
        ["Auto-detect", "Fine resolution (Rayleigh fingerprint)",
         "Coarse resolution (multi-metric voting)"],
        index=0,
    )

    input_method = st.radio(
        "Input method",
        ["Upload ZIP", "Browse files", "Folder path"],
        index=0,
        horizontal=True,
    )

    uploaded_files = None
    uploaded_zip = None
    folder_path = None

    if input_method == "Upload ZIP":
        uploaded_zip = st.file_uploader(
            "Drop a ZIP of SOR files here",
            type=["zip"],
            accept_multiple_files=False,
            key=f"zip_upload_{st.session_state.upload_key}",
        )
        if uploaded_zip:
            st.caption(f"ZIP: {uploaded_zip.name} ({uploaded_zip.size / 1024:.0f} KB)")
    elif input_method == "Browse files":
        uploaded_files = st.file_uploader(
            "Drop SOR files here",
            type=["sor"],
            accept_multiple_files=True,
            key=f"upload_{st.session_state.upload_key}",
        )
    else:
        folder_path = st.text_input(
            "Paste or drag folder path here",
            value=st.session_state.get("folder_path", ""),
            placeholder="/Users/you/Desktop/My Traces/",
        )
        if folder_path:
            folder_path = folder_path.strip().strip("'\"")
            st.session_state.folder_path = folder_path
        if folder_path and os.path.isdir(folder_path):
            from rayleigh_fingerprint import collect_sor_files as _collect
            _found = _collect([folder_path])
            st.caption(f"Found {len(_found)} .sor files")
        elif folder_path:
            st.warning("Folder not found")

    has_input = (bool(uploaded_files) or bool(uploaded_zip) or
                 (folder_path and os.path.isdir(folder_path)))

    if st.button("Clear All", use_container_width=True):
        old_key = st.session_state.upload_key
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.upload_key = old_key + 1
        st.rerun()

    route_name = st.text_input("Route name (optional)", "")

    with st.expander("Advanced Settings"):
        tier1 = st.number_input("Tier 1 threshold (dB)", value=EXACT_DUP_THRESHOLD,
                                format="%.4f", step=0.001)
        tier2 = st.number_input("Tier 2 threshold (dB)", value=PROBABLE_DUP_THRESHOLD,
                                format="%.4f", step=0.005)
        event_buffer = st.number_input("Event buffer (samples)", value=DEFAULT_EVENT_BUFFER,
                                       min_value=0, max_value=100, step=5)
        vote_pct = st.number_input("Vote % (coarse)", value=0.5, min_value=0.1,
                                   max_value=5.0, step=0.1, format="%.1f") / 100.0
        min_votes = st.number_input("Min votes (coarse)", value=5, min_value=1, max_value=7)

    run_button = st.button("Run Analysis", type="primary", use_container_width=True,
                           disabled=not has_input)


# ── Helper: write uploaded files to temp dir ─────────────────────────────────

def stage_files(uploaded, prefix="sor_"):
    """Write UploadedFile objects to a temp directory. Return list of paths."""
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    paths = []
    for uf in uploaded:
        fp = os.path.join(tmpdir, uf.name)
        with open(fp, 'wb') as f:
            f.write(uf.getbuffer())
        paths.append(fp)
    return sorted(paths), tmpdir


def stage_zip(uploaded_zip, prefix="sor_zip_"):
    """Extract SOR files from a ZIP to a temp directory. Return list of paths."""
    import zipfile
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    paths = []
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getbuffer()), 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith('.sor') and not name.startswith('__MACOSX'):
                # Flatten into tmpdir (ignore subdirectory structure)
                basename = os.path.basename(name)
                if not basename:
                    continue
                fp = os.path.join(tmpdir, basename)
                with zf.open(name) as src, open(fp, 'wb') as dst:
                    dst.write(src.read())
                paths.append(fp)
    return sorted(paths), tmpdir


def detect_resolution(filepaths):
    """Auto-detect fine vs coarse by checking segment count."""
    try:
        from rayleigh_fingerprint import extract_fingerprint
        sor = parse_sor(filepaths[0])
        rfp = extract_fingerprint(sor)
        return "fine" if rfp.num_segments > 1 else "coarse"
    except Exception:
        return "fine"


def render_pdf(html_content, pdf_path):
    """Render HTML to PDF. Try Chrome first, fall back to xhtml2pdf."""
    # Try Chrome headless first (best quality)
    chrome = _find_chrome()
    if chrome:
        tmpdir = tempfile.mkdtemp(prefix="rayleigh_pdf_")
        html_path = os.path.join(tmpdir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        try:
            result = subprocess.run(
                [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
                 '--run-all-compositor-stages-before-draw',
                 '--virtual-time-budget=5000',
                 f'--print-to-pdf={os.path.abspath(pdf_path)}',
                 '--print-to-pdf-no-header', '--no-pdf-header-footer',
                 'file://' + os.path.abspath(html_path)],
                capture_output=True, timeout=120)
            if result.returncode == 0 and os.path.exists(pdf_path):
                return True
        except Exception:
            pass

    # Fall back to xhtml2pdf (pure Python, no system deps)
    try:
        from xhtml2pdf import pisa
        with open(pdf_path, 'wb') as f:
            result = pisa.CreatePDF(html_content, dest=f)
        return not result.err and os.path.exists(pdf_path)
    except ImportError:
        return False
    except Exception:
        return False


# ── Run analysis ─────────────────────────────────────────────────────────────

if run_button and has_input:
    # Get file paths — from ZIP, uploads, or folder path
    if folder_path and os.path.isdir(folder_path):
        from rayleigh_fingerprint import collect_sor_files as _collect
        filepaths = _collect([folder_path])
        st.toast(f"Found {len(filepaths)} SOR files")
    elif uploaded_zip:
        progress_bar = st.progress(0, text="Extracting ZIP...")
        filepaths, tmpdir = stage_zip(uploaded_zip)
        progress_bar.progress(1.0, text=f"Extracted {len(filepaths)} SOR files from ZIP")
        progress_bar.empty()
    else:
        # Stage uploaded files with progress bar
        progress_bar = st.progress(0, text="Staging uploaded files...")
        tmpdir = tempfile.mkdtemp(prefix="rayleigh_fwd_")
        filepaths = []
        total = len(uploaded_files)
        for i, uf in enumerate(uploaded_files):
            fp = os.path.join(tmpdir, uf.name)
            with open(fp, 'wb') as f:
                f.write(uf.getbuffer())
            filepaths.append(fp)
            progress_bar.progress((i + 1) / total,
                                  text=f"Staging files... {i+1}/{total}")
        filepaths.sort()
        progress_bar.empty()

    # Detect mode
    if mode == "Auto-detect":
        resolution = detect_resolution(filepaths)
    elif mode == "Fine resolution (Rayleigh fingerprint)":
        resolution = "fine"
    elif mode == "Coarse resolution (multi-metric voting)":
        resolution = "coarse"
    else:
        resolution = "fine"

    log_buf = io.StringIO()
    rname = route_name or "report"
    summary_lines = []

    analysis_bar = st.progress(0, text=f"Analyzing {len(filepaths)} traces...")

    if resolution == "fine":
        analysis_bar.progress(5, text=f"Running Rayleigh fingerprint on {len(filepaths)} traces...")
        with redirect_stdout(log_buf):
            results, event_meta = run_batch(
                filepaths, tier1_thresh=tier1, tier2_thresh=tier2,
                event_buffer=event_buffer)

        analysis_bar.progress(50, text="Processing results...")
        reliable = [r for r in results if r.tier_label != "insufficient data"
                    and r.rayleigh_rmse < float('inf')]
        tier1_pairs = [r for r in results if r.tier == 1]
        tier2_pairs = [r for r in results if r.tier == 2]

        summary_lines.append(f"**Mode:** Fine resolution (Rayleigh fingerprint)")
        summary_lines.append(f"**Files:** {len(filepaths)} traces")
        summary_lines.append(f"**Thresholds:** Tier 1 = {tier1*1000:.1f} mdB, Tier 2 = {tier2*1000:.1f} mdB")
        summary_lines.append(f"**Reliable pairs:** {len(reliable):,}")
        summary_lines.append(f"**Exact duplicates:** {len(tier1_pairs)}")
        summary_lines.append(f"**Probable duplicates:** {len(tier2_pairs)}")
        if tier1_pairs:
            for r in tier1_pairs:
                summary_lines.append(f"  - ★ {r.fiber_a} ↔ {r.fiber_b} ({r.rayleigh_rmse*1000:.3f} mdB)")
        if tier2_pairs:
            for r in tier2_pairs:
                summary_lines.append(f"  - ∼ {r.fiber_a} ↔ {r.fiber_b} ({r.rayleigh_rmse*1000:.3f} mdB)")
        if reliable:
            summary_lines.append(f"**Closest pair:** {reliable[0].fiber_a} ↔ {reliable[0].fiber_b} "
                                 f"({reliable[0].rayleigh_rmse*1000:.3f} mdB)")
        summary_lines.append(f"**Pairs in PDF:** {min(10, len(tier1_pairs) + len(tier2_pairs) + 2)}")

        analysis_bar.progress(60, text="Generating distribution chart...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        chart_png_path = None
        all_rmse = np.array([r.rayleigh_rmse * 1000 for r in reliable
                             if r.rayleigh_rmse < float('inf')])
        if len(all_rmse):
            chart_tmpdir = tempfile.mkdtemp(prefix="rayleigh_chart_")
            chart_png_path = os.path.join(chart_tmpdir, "distribution.png")
            fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})
            ax = axes[0]
            bin_width = max(1, (max(all_rmse) - min(all_rmse)) / 80)
            bins = np.arange(0, max(all_rmse) + bin_width, bin_width)
            counts, edges, patches = ax.hist(all_rmse, bins=bins,
                                              color='#4A90D9', edgecolor='white',
                                              linewidth=0.5, alpha=0.85)
            for patch, left_edge in zip(patches, edges[:-1]):
                if left_edge < tier1 * 1000:
                    patch.set_facecolor('#E34B4A'); patch.set_alpha(0.9)
                elif left_edge < tier2 * 1000:
                    patch.set_facecolor('#E0C060'); patch.set_alpha(0.9)
            ax.axvline(x=tier1 * 1000, color='#E34B4A', linestyle='--',
                       linewidth=1.5, label=f'Tier 1 ({tier1*1000:.1f} mdB)')
            ax.axvline(x=tier2 * 1000, color='#E0C060', linestyle='--',
                       linewidth=1.5, label=f'Tier 2 ({tier2*1000:.1f} mdB)')
            # Annotate all tier 1 duplicates on the chart
            dup_results = [r for r in reliable if r.tier == 1]
            for di, r0 in enumerate(dup_results):
                y_offset = max(counts) * (0.4 + di * 0.15)
                ax.annotate(
                    f'{r0.fiber_a} \u2194 {r0.fiber_b}\n{r0.rayleigh_rmse*1000:.1f} mdB',
                    xy=(r0.rayleigh_rmse * 1000, 1),
                    xytext=(r0.rayleigh_rmse * 1000 + max(all_rmse) * 0.05, y_offset),
                    fontsize=9, fontweight='bold', color='#E34B4A',
                    arrowprops=dict(arrowstyle='->', color='#E34B4A', lw=1.5))
            ax.set_xlabel('Rayleigh RMSE (mdB)', fontsize=12)
            ax.set_ylabel('Number of pairs', fontsize=12)
            rn = route_name or 'All Traces'
            ax.set_title(f"Rayleigh's Fingerprint \u2014 RMSE Distribution ({rn})",
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.set_xlim(0, max(all_rmse) + max(all_rmse) * 0.05)
            stats_text = (f'Pairs: {len(all_rmse)}\nMean: {np.mean(all_rmse):.1f} mdB\n'
                          f'Min: {np.min(all_rmse):.1f} mdB\nMax: {np.max(all_rmse):.1f} mdB')
            ax.text(0.98, 0.65, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='#ccc', alpha=0.9), fontfamily='monospace')
            ax2 = axes[1]
            colors = ['#E34B4A' if v < tier1 * 1000 else '#E0C060' if v < tier2 * 1000
                      else '#4A90D9' for v in all_rmse]
            ax2.scatter(all_rmse, np.zeros_like(all_rmse), c=colors, s=12, alpha=0.6, edgecolors='none')
            ax2.axvline(x=tier1 * 1000, color='#E34B4A', linestyle='--', linewidth=1.0)
            ax2.axvline(x=tier2 * 1000, color='#E0C060', linestyle='--', linewidth=1.0)
            ax2.set_xlabel('Rayleigh RMSE (mdB)', fontsize=11)
            ax2.set_yticks([])
            ax2.set_xlim(ax.get_xlim())
            ax2.set_title('Individual pair RMSE values', fontsize=11, color='#888')
            plt.tight_layout()
            plt.savefig(chart_png_path, dpi=150, bbox_inches='tight')
            plt.close()

        analysis_bar.progress(80, text="Generating PDF HTML...")
        # Show all flagged pairs (tier1 + tier2), then pad to 10 with closest
        pairs_to_show = []
        flagged_set = set()
        for r in reliable:
            if r.tier in (1, 2):
                pairs_to_show.append(r)
                flagged_set.add((r.fiber_a, r.fiber_b))
        for r in reliable:
            if len(pairs_to_show) >= 10:
                break
            if (r.fiber_a, r.fiber_b) not in flagged_set:
                pairs_to_show.append(r)

        html = generate_pdf_html(
            pairs_to_show, event_meta, tier1, tier2,
            len(event_meta) or len(filepaths), len(results),
            route_name, chart_png_path=chart_png_path)

    elif resolution == "coarse":
        analysis_bar.progress(5, text=f"Running multi-metric voting on {len(filepaths)} traces...")
        with redirect_stdout(log_buf):
            flagged, all_results, event_meta, vote_counts = run_voting(
                filepaths, vote_pct=vote_pct, min_votes=min_votes)

        analysis_bar.progress(60, text="Processing suspects...")
        reliable = [r for r in all_results if r.tier_label != "insufficient data"
                    and r.rayleigh_rmse < float('inf')]
        n_flagged = len(flagged) if flagged else 0

        summary_lines.append(f"**Mode:** Coarse resolution (multi-metric voting)")
        summary_lines.append(f"**Files:** {len(filepaths)} traces")
        summary_lines.append(f"**Reliable pairs:** {len(reliable):,}")
        summary_lines.append(f"**Suspects:** {n_flagged}")

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

            flagged.sort(key=lambda x: x[3].get('max_splice_diff_mdB', 999))
            splice_vals = [c[3].get('max_splice_diff_mdB', 999) for c in flagged
                           if c[3].get('max_splice_diff_mdB', 999) < float('inf')]
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
                for votes, a, b, rd in flagged:
                    spl = rd.get('max_splice_diff_mdB', 999)
                    if spl <= splice_threshold: rd['confidence'] = 'HIGH'
                    elif spl < splice_threshold * 2: rd['confidence'] = 'REVIEW'
                    else: rd['confidence'] = ''
            else:
                for votes, a, b, rd in flagged:
                    spl = rd.get('max_splice_diff_mdB', 999)
                    if spl < 25: rd['confidence'] = 'HIGH'
                    elif spl < 50: rd['confidence'] = 'REVIEW'
                    else: rd['confidence'] = ''

            for v, a, b, rd in flagged:
                conf = rd.get('confidence', '')
                conf_str = f" [{conf}]" if conf else ""
                summary_lines.append(f"  - {a} \u2194 {b} ({v}/{len(METRIC_NAMES)} votes){conf_str}")

        analysis_bar.progress(80, text="Building dashboard...")
        html = build_dashboard(flagged, all_results, event_meta,
                               vote_counts, vote_pct, min_votes,
                               route_name or "Coarse Analysis")

    analysis_bar.progress(90, text="Rendering PDF...")

    st.session_state.log_output = log_buf.getvalue()
    st.session_state.html_report = html
    st.session_state.html_name = f"rayleigh_{rname}.html"
    st.session_state.summary = "\n\n".join(summary_lines)

    st.session_state.summary = "\n\n".join(summary_lines)

    # Auto-generate PDF
    pdf_tmpdir = tempfile.mkdtemp(prefix="rayleigh_pdf_")
    pdf_path = os.path.join(pdf_tmpdir, f"rayleigh_{rname}.pdf")
    success = render_pdf(html, pdf_path)
    if success:
        with open(pdf_path, 'rb') as f:
            st.session_state.pdf_bytes = f.read()
        st.session_state.pdf_name = f"rayleigh_{rname}.pdf"
    else:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None

    analysis_bar.progress(100, text="Done!")
    analysis_bar.empty()

    st.session_state.analysis_done = True


# ── Display results ──────────────────────────────────────────────────────────

if st.session_state.get("analysis_done"):

    st.markdown("""
    <div class="tc-hero">
        <h1>Analysis Complete</h1>
        <p>Rayleigh backscatter fingerprinting - duplicate trace detection</p>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    st.markdown("<br>", unsafe_allow_html=True)

    # Download buttons
    col1, col2 = st.columns(2)

    if st.session_state.pdf_bytes:
        with col1:
            st.download_button(
                "⬇ Download PDF Report",
                st.session_state.pdf_bytes,
                file_name=st.session_state.pdf_name,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

    if st.session_state.html_report:
        with col2:
            st.download_button(
                "⬇ Download HTML Report",
                st.session_state.html_report,
                file_name=st.session_state.html_name,
                mime="text/html",
                use_container_width=True,
            )

    if not st.session_state.pdf_bytes:
        st.warning("PDF generation failed — Chrome not found. Use the HTML report instead.")

    # Analysis log (collapsed)
    with st.expander("Analysis Log"):
        st.code(st.session_state.log_output or "No log output.", language=None)

else:
    st.markdown("""
    <div class="tc-hero">
        <h1>Rayleigh Duplicate<br>Detector</h1>
        <p>OTDR trace duplicate detection using Rayleigh backscatter fingerprinting</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; gap:18px; align-items:stretch; margin-bottom:18px;">
        <div class="tc-card" style="flex:1; margin-bottom:0;">
            <div class="tc-card-title">Fine Resolution - Rayleigh Fingerprint</div>
            <ul class="tc-list">
                <li>Compares Rayleigh backscatter segments between fiber pairs</li>
                <li>Tier 1 - exact duplicates below tight RMSE threshold</li>
                <li>Tier 2 - probable duplicates below looser threshold</li>
                <li>Best for trimmed, high-resolution OTDR traces</li>
            </ul>
        </div>
        <div class="tc-card" style="flex:1; margin-bottom:0;">
            <div class="tc-card-title">Coarse Resolution - Multi-Metric Voting</div>
            <ul class="tc-list">
                <li>Votes across 7 shape metrics per fiber pair</li>
                <li>Catches duplicates when Rayleigh segments are too short</li>
                <li>Auto-mode selects fine or coarse based on trace resolution</li>
                <li>Best for untrimmed or short-segment traces</li>
            </ul>
        </div>
    </div>
    <div style="display:flex; gap:18px; align-items:stretch; margin-bottom:18px;">
        <div class="tc-card" style="flex:1; margin-bottom:0;">
            <div class="tc-card-title">How To Use</div>
            <ul class="tc-list">
                <li>Upload SOR files as a ZIP or browse individual files</li>
                <li>Select Auto-detect mode or choose Fine/Coarse manually</li>
                <li>Enter a route name (optional) for the report filename</li>
                <li>Click <strong>Run Analysis</strong> to download the PDF report</li>
            </ul>
        </div>
        <div class="tc-card" style="flex:1; margin-bottom:0;">
            <div class="tc-card-title">Output</div>
            <ul class="tc-list">
                <li>PDF report with RMSE distribution chart and flagged pairs</li>
                <li>HTML report for browser viewing</li>
                <li>Tier 1 (exact) and Tier 2 (probable) duplicate classifications</li>
                <li>Advanced settings available for threshold tuning</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
