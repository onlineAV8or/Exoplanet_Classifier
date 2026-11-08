import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"

st.set_page_config(page_title="KEPLER // Exoplanet Classifier",
                   page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Holographic / sci-fi theme (CSS only — no behavioural change).
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
:root{
  --cyan:#22d3ee; --cyan-soft:#67e8f9; --magenta:#e879f9; --amber:#fbbf24;
  --ink:#05060f; --panel:rgba(12,20,45,.55); --line:rgba(103,232,249,.35);
  --text:#dbeafe; --muted:#8aa0c8;
}
/* ---- deep space backdrop ---- */
.stApp{ background:
    radial-gradient(1200px 600px at 80% -10%, rgba(232,121,249,.10), transparent 60%),
    radial-gradient(900px 500px at 5% 10%, rgba(34,211,238,.10), transparent 55%),
    linear-gradient(180deg,#05060f 0%, #070a1a 50%, #04030a 100%);
}
#space-bg{ position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
#space-bg .stars{ position:absolute; inset:-50%; width:200%; height:200%;
  background-image:
    radial-gradient(1.4px 1.4px at 20% 30%, #fff, transparent),
    radial-gradient(1.4px 1.4px at 75% 15%, #cfe9ff, transparent),
    radial-gradient(1px 1px at 50% 60%, #fff, transparent),
    radial-gradient(1px 1px at 12% 80%, #bae6fd, transparent),
    radial-gradient(1.6px 1.6px at 88% 70%, #fff, transparent),
    radial-gradient(1px 1px at 35% 45%, #e0f2fe, transparent),
    radial-gradient(1px 1px at 65% 90%, #fff, transparent);
  background-size:280px 280px; animation:drift 140s linear infinite; opacity:.9; }
#space-bg .stars.two{ background-size:420px 420px; animation-duration:240s;
  animation-direction:reverse; filter:blur(.3px); opacity:.6; }
#space-bg .twinkle{ position:absolute; inset:0;
  background-image:
    radial-gradient(1.5px 1.5px at 30% 20%, #fff, transparent),
    radial-gradient(2px 2px at 60% 70%, #a5f3fc, transparent),
    radial-gradient(1.5px 1.5px at 85% 40%, #fff, transparent);
  background-size:300px 300px; animation:tw 4s ease-in-out infinite; }
@keyframes drift{ to{ transform:translate(-140px,-90px); } }
@keyframes tw{ 0%,100%{opacity:.25} 50%{opacity:.9} }

/* keep real content above the starfield */
section.main > div.block-container{ position:relative; z-index:1; max-width:1180px; }
.stApp [data-testid="stHeader"]{ background:transparent; }
.stApp, .stMarkdown, p, span, label, li{ color:var(--text); font-family:'Rajdhani',sans-serif; }

/* ---- hero ---- */
.hero{ position:relative; z-index:1; display:flex; align-items:center; gap:34px;
  padding:26px 30px; margin:4px 0 10px; border-radius:18px;
  background:linear-gradient(135deg, rgba(10,16,40,.75), rgba(8,12,30,.55));
  border:1px solid var(--line); overflow:hidden;
  box-shadow:0 0 42px rgba(34,211,238,.12), inset 0 0 60px rgba(34,211,238,.05); }
.hero h1{ font-family:'Orbitron',sans-serif; font-weight:900; letter-spacing:4px;
  font-size:2.3rem; margin:0; color:#eaf6ff;
  text-shadow:0 0 18px rgba(34,211,238,.55), 0 0 42px rgba(34,211,238,.25); }
.hero .sub{ color:var(--cyan-soft); font-family:'Share Tech Mono',monospace;
  letter-spacing:2px; font-size:.92rem; margin-top:6px; }
.hero .tag{ color:var(--muted); font-size:.86rem; margin-top:8px; max-width:640px; }

/* orbiting planet */
.sys{ position:relative; width:130px; height:130px; flex:0 0 130px; }
.planet{ position:absolute; top:50%; left:50%; width:74px; height:74px;
  transform:translate(-50%,-50%); border-radius:50%;
  background:radial-gradient(circle at 32% 28%, #a5f3fc 0%, #22d3ee 26%, #1d4ed8 62%, #111a3d 100%);
  box-shadow:0 0 26px rgba(34,211,238,.75), inset -14px -10px 30px rgba(0,0,0,.55),
             inset 8px 6px 18px rgba(165,243,252,.35); }
.ring{ position:absolute; top:50%; left:50%; width:118px; height:118px;
  transform:translate(-50%,-50%) rotateX(74deg); border-radius:50%;
  border:2px solid rgba(103,232,249,.45); box-shadow:0 0 14px rgba(34,211,238,.35); }
.moon{ position:absolute; top:50%; left:50%; width:124px; height:124px;
  transform:translate(-50%,-50%); animation:spin 7s linear infinite; }
.moon::after{ content:""; position:absolute; top:-4px; left:50%; width:11px; height:11px;
  border-radius:50%; background:#fde68a; box-shadow:0 0 12px #fbbf24; transform:translateX(-50%); }
@keyframes spin{ to{ transform:translate(-50%,-50%) rotate(360deg); } }

/* ---- HUD status strip ---- */
.hud{ display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 18px; position:relative; z-index:1; }
.hud .chip{ font-family:'Share Tech Mono',monospace; font-size:.8rem; letter-spacing:1px;
  color:var(--cyan-soft); padding:7px 14px; border-radius:999px;
  border:1px solid var(--line); background:rgba(34,211,238,.06); }
.hud .chip b{ color:#fff; }
.hud .live{ color:#34d399; border-color:rgba(52,211,153,.4); background:rgba(52,211,153,.08); }
.hud .live::before{ content:"● "; animation:tw 1.6s infinite; }

/* ---- holographic panels ---- */
.holo{ position:relative; z-index:1; border-radius:16px; padding:20px 22px; margin:10px 0 22px;
  background:var(--panel); border:1px solid var(--line);
  backdrop-filter:blur(9px); -webkit-backdrop-filter:blur(9px);
  box-shadow:0 0 34px rgba(34,211,238,.10), inset 0 0 40px rgba(34,211,238,.04);
  background-image:repeating-linear-gradient(0deg, rgba(34,211,238,.035) 0 2px, transparent 2px 5px); }
.holo::before,.holo::after{ content:""; position:absolute; width:18px; height:18px; }
.holo::before{ top:8px; left:8px; border-top:2px solid var(--cyan); border-left:2px solid var(--cyan); }
.holo::after{ bottom:8px; right:8px; border-bottom:2px solid var(--magenta); border-right:2px solid var(--magenta); }
.sect{ font-family:'Orbitron',sans-serif; letter-spacing:3px; color:#eaf6ff; font-size:1rem;
  text-transform:uppercase; margin:6px 0 2px;
  text-shadow:0 0 14px rgba(34,211,238,.4); }
.sect::before{ content:"▸ "; color:var(--cyan); }

/* ---- holo comparison table ---- */
.holo-table{ width:100%; border-collapse:separate; border-spacing:0 6px; font-family:'Rajdhani',sans-serif; }
.holo-table th{ font-family:'Share Tech Mono',monospace; font-size:.74rem; letter-spacing:1px;
  color:var(--muted); text-transform:uppercase; text-align:right; padding:4px 12px; }
.holo-table th:first-child{ text-align:left; }
.holo-table td{ padding:10px 12px; text-align:right; font-size:1rem; color:#eaf6ff;
  background:rgba(12,20,45,.5); border-top:1px solid rgba(103,232,249,.12);
  border-bottom:1px solid rgba(103,232,249,.12); }
.holo-table td:first-child{ text-align:left; font-weight:600; letter-spacing:.5px;
  border-left:3px solid transparent; border-radius:8px 0 0 8px; }
.holo-table td:last-child{ border-radius:0 8px 8px 0; }
.holo-table tr.win td{ background:rgba(34,211,238,.12); }
.holo-table tr.win td:first-child{ border-left-color:var(--cyan);
  text-shadow:0 0 12px rgba(34,211,238,.6); }
.win-badge{ color:#05060f; background:var(--cyan); font-family:'Share Tech Mono',monospace;
  font-size:.62rem; padding:2px 7px; border-radius:6px; margin-left:8px; letter-spacing:1px;
  box-shadow:0 0 12px rgba(34,211,238,.6); }

/* ---- neon stat tiles ---- */
.tiles{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-top:6px; }
.tile{ position:relative; border-radius:14px; padding:16px 14px; text-align:center;
  background:linear-gradient(160deg, rgba(34,211,238,.10), rgba(12,20,45,.5));
  border:1px solid var(--line); box-shadow:inset 0 0 22px rgba(34,211,238,.06); }
.tile .k{ font-family:'Share Tech Mono',monospace; font-size:.68rem; letter-spacing:1.5px;
  color:var(--muted); text-transform:uppercase; }
.tile .v{ font-family:'Orbitron',sans-serif; font-weight:700; font-size:1.5rem; margin-top:6px;
  color:#eaf6ff; text-shadow:0 0 16px rgba(34,211,238,.55); }
@media (max-width:900px){ .tiles{ grid-template-columns:repeat(3,1fr); } }

/* ---- widgets ---- */
[data-testid="stFileUploaderDropzone"]{ background:rgba(12,20,45,.5);
  border:1.5px dashed var(--line); border-radius:14px; }
.scan-label{ font-family:'Share Tech Mono',monospace; letter-spacing:2px; color:#67e8f9;
  font-size:.82rem; text-transform:uppercase; margin:4px 0 6px;
  text-shadow:0 0 12px rgba(34,211,238,.45); }
.scan-label b{ color:#eaf6ff; }
.stSelectbox div[data-baseweb="select"] > div{
  background:linear-gradient(135deg, rgba(34,211,238,.14), rgba(12,20,45,.72)) !important;
  border:1.5px solid var(--cyan) !important; border-radius:12px !important;
  min-height:48px; font-size:1.02rem;
  box-shadow:0 0 16px rgba(34,211,238,.30), inset 0 0 16px rgba(34,211,238,.06);
  animation:selglow 2.2s ease-in-out infinite; }
.stSelectbox div[data-baseweb="select"] > div:hover{ border-color:#a5f3fc !important; }
.stSelectbox [data-baseweb="select"] *{ color:#eaf6ff !important; }
.stSelectbox svg{ fill:#22d3ee !important; color:#22d3ee !important; }
@keyframes selglow{
  0%,100%{ box-shadow:0 0 14px rgba(34,211,238,.28), inset 0 0 14px rgba(34,211,238,.05); }
  50%{ box-shadow:0 0 26px rgba(34,211,238,.58), inset 0 0 18px rgba(34,211,238,.11); } }
ul[role="listbox"]{ background:#0b1026 !important; border:1px solid var(--line) !important;
  box-shadow:0 0 22px rgba(34,211,238,.25) !important; }
li[role="option"]{ color:#dbeafe !important; }
li[role="option"]:hover{ background:rgba(34,211,238,.16) !important; }
li[role="option"][aria-selected="true"]{ background:rgba(34,211,238,.26) !important; }
.stButton button, .stDownloadButton button{ font-family:'Share Tech Mono',monospace; letter-spacing:1px;
  background:linear-gradient(135deg, rgba(34,211,238,.18), rgba(232,121,249,.14));
  color:#eaf6ff; border:1px solid var(--line); border-radius:10px;
  box-shadow:0 0 18px rgba(34,211,238,.18); transition:.2s; }
.stButton button:hover, .stDownloadButton button:hover{ box-shadow:0 0 26px rgba(34,211,238,.4);
  border-color:var(--cyan); }
hr{ border-color:rgba(103,232,249,.18); }
/* ---- left navigation sidebar ---- */
[data-testid="stSidebar"]{
  background:linear-gradient(160deg, rgba(11,16,38,.97), rgba(6,9,22,.97)) !important;
  border-right:1px solid var(--line); box-shadow:0 0 40px rgba(34,211,238,.12); }
[data-testid="stSidebar"] .navtitle{ font-family:'Orbitron',sans-serif; letter-spacing:3px;
  color:#eaf6ff; font-size:.95rem; margin:4px 0 14px; text-shadow:0 0 14px rgba(34,211,238,.45); }
[data-testid="stSidebar"] .navfoot{ margin-top:14px; color:#8aa0c8; font-size:.78rem;
  font-family:'Rajdhani',sans-serif; line-height:1.4; }
[data-testid="stSidebar"] .navlocked{ margin:10px 0 2px; padding:9px 11px; border-radius:8px;
  font-family:'Share Tech Mono',monospace; font-size:.74rem; letter-spacing:.3px; line-height:1.4;
  color:#fde68a; background:rgba(251,191,36,.10); border:1px solid rgba(251,191,36,.42);
  box-shadow:0 0 14px rgba(251,191,36,.18); }
[data-testid="stSidebar"] [class*="st-key-navlock_"] button{ opacity:.55;
  border:1px dashed rgba(251,191,36,.45) !important; color:#cbb784 !important;
  background:rgba(251,191,36,.05) !important; }
[data-testid="stSidebar"] [class*="st-key-navlock_"] button:hover{ opacity:.85; transform:none;
  border-color:rgba(251,191,36,.75) !important; color:#fde68a !important;
  background:rgba(251,191,36,.10) !important; }
[data-testid="stSidebar"] .stButton button{ justify-content:flex-start; text-align:left;
  font-family:'Rajdhani',sans-serif; font-size:1rem; letter-spacing:.5px; font-weight:500;
  background:rgba(34,211,238,.05); border:1px solid transparent; color:#cfe3ff;
  box-shadow:none; border-radius:9px; padding:9px 12px; transition:.15s; }
[data-testid="stSidebar"] .stButton button:hover{ background:rgba(34,211,238,.14);
  border-color:var(--line); color:#eaf6ff; transform:translateX(3px); box-shadow:none; }
@media (min-width:821px){
  [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"]{ display:none !important; } }
html{ scroll-behavior:smooth; }
#diagnostics{ display:block; position:relative; scroll-margin-top:16px; }
/* glowing "jump to diagnostics" call-to-action */
.jump-btn{ display:inline-block; margin:16px 0 4px; padding:13px 30px; border-radius:12px;
  font-family:'Share Tech Mono',monospace; letter-spacing:2px; font-size:.92rem; font-weight:700;
  text-decoration:none !important; color:#05060f !important; cursor:pointer;
  background:linear-gradient(135deg,#22d3ee,#67e8f9); border:1px solid #a5f3fc;
  animation:pulseglow 1.8s ease-in-out infinite; transition:transform .15s ease; }
.jump-btn:hover{ transform:translateY(2px);
  box-shadow:0 0 30px rgba(34,211,238,.85), 0 0 64px rgba(232,121,249,.4); }
@keyframes pulseglow{
  0%,100%{ box-shadow:0 0 16px rgba(34,211,238,.45), 0 0 34px rgba(34,211,238,.20); }
  50%{ box-shadow:0 0 30px rgba(34,211,238,.80), 0 0 62px rgba(34,211,238,.42); } }
.anchor{ display:block; position:relative; scroll-margin-top:16px; }
/* ---- burger navigation (pure-CSS :target drawer) ---- */
.burger{ position:fixed; top:12px; left:12px; z-index:4100; width:44px; height:44px;
  display:flex; align-items:center; justify-content:center; font-size:1.35rem;
  text-decoration:none !important; color:#eaf6ff !important; border-radius:11px;
  border:1px solid var(--line); background:rgba(10,16,40,.72);
  backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
  box-shadow:0 0 16px rgba(34,211,238,.3); transition:.2s; }
.burger:hover{ border-color:var(--cyan); box-shadow:0 0 26px rgba(34,211,238,.6); }
.navwrap{ position:fixed; inset:0; z-index:4000; visibility:hidden; }
.navwrap .navback{ position:absolute; inset:0; background:rgba(3,5,12,.62);
  opacity:0; transition:opacity .3s ease; }
.navpanel{ position:absolute; top:0; left:0; height:100%; width:270px; padding:22px 16px;
  transform:translateX(-100%); transition:transform .32s cubic-bezier(.4,0,.2,1);
  background:linear-gradient(160deg, rgba(11,16,38,.98), rgba(6,9,22,.98));
  border-right:1px solid var(--line); box-shadow:0 0 50px rgba(34,211,238,.25); overflow-y:auto; }
#nav:target{ visibility:visible; }
#nav:target .navback{ opacity:1; }
#nav:target .navpanel{ transform:translateX(0); }
.navpanel .navhead{ font-family:'Orbitron',sans-serif; letter-spacing:3px; color:#eaf6ff;
  font-size:.9rem; margin:28px 0 14px; text-shadow:0 0 14px rgba(34,211,238,.4); }
.navpanel a.navlink{ display:block; text-decoration:none !important; color:#cfe3ff !important;
  font-family:'Rajdhani',sans-serif; font-size:1.02rem; letter-spacing:.5px; padding:11px 12px;
  margin:4px 0; border-radius:9px; border:1px solid transparent; transition:.15s; }
.navpanel a.navlink::before{ content:"▸ "; color:var(--cyan); }
.navpanel a.navlink:hover{ background:rgba(34,211,238,.12); border-color:var(--line);
  color:#eaf6ff !important; transform:translateX(3px); }
.navclose{ position:absolute; top:14px; right:14px; text-decoration:none !important;
  color:#8aa0c8 !important; font-family:'Share Tech Mono',monospace; font-size:.74rem;
  border:1px solid var(--line); padding:5px 10px; border-radius:8px; }
.navclose:hover{ color:#eaf6ff !important; border-color:var(--cyan); }
/* ---- responsive / mobile ---- */
@media (max-width:640px){
  section.main > div.block-container{ padding-left:10px; padding-right:10px; }
  .hero{ flex-direction:column; text-align:center; gap:14px; padding:20px 16px; }
  .hero .sys{ margin:0 auto; }
  .hero h1{ font-size:1.5rem; letter-spacing:2px; }
  .hero .sub{ font-size:.82rem; }
  .hero .tag{ font-size:.82rem; }
  .hud{ gap:7px; }
  .hud .chip{ font-size:.72rem; padding:6px 10px; }
  .sect{ font-size:.9rem; letter-spacing:2px; }
  .holo{ padding:16px 14px; }
  .tiles{ grid-template-columns:repeat(2,1fr) !important; gap:9px; }
  .tile .v{ font-size:1.2rem; }
  #boot h2{ font-size:1.25rem; letter-spacing:2px; }
  #boot .log{ min-width:0; width:86%; font-size:.76rem; }
  #boot .bar{ width:80%; }
  .navpanel{ width:82%; }
  .hero h1{ padding-left:40px; }
}
#MainMenu, footer{ visibility:hidden; }
</style>
<div id="space-bg"><div class="stars"></div><div class="stars two"></div><div class="twinkle"></div></div>
""", unsafe_allow_html=True)

# ---- persistent left navigation (native sidebar; scrolls to sections by header text) ----
def _scroll_js(prefix):
    return ("<script>(function(){var p=window.parent.document;"
            "function go(x,n){var els=[].slice.call(p.querySelectorAll('.sect'));"
            "var t=els.filter(function(e){return e.textContent.trim().toLowerCase()"
            ".indexOf(x.toLowerCase())===0;})[0];"
            "if(t){t.scrollIntoView({behavior:'smooth',block:'start'});}"
            "else if(n>0){setTimeout(function(){go(x,n-1);},120);}}"
            "go(" + json.dumps(prefix) + ",18);})();</script>")

SAMPLE_PATH = BASE_DIR / "test_data.csv"
OPT_UPLOAD = "Upload my CSV"
OPT_SAMPLE = "Use bundled sample"


def data_is_loaded():
    """True if a test set is selected — read from persisted widget state so the
    sidebar (rendered before the uploader) reflects the current choice."""
    if st.session_state.get("src_choice", OPT_UPLOAD) == OPT_SAMPLE and SAMPLE_PATH.exists():
        return True
    return st.session_state.get("uploader") is not None


# (label, section-header prefix, requires_data)
NAV_ITEMS = [
    ("Telemetry Uplink", "Telemetry Uplink", False),
    ("Mission Objective", "Mission Objective", False),
    ("KOI Field Guide", "KOI Field Guide", False),
    ("Dataset Dossier", "Dataset Dossier", False),
    ("Fleet Diagnostics", "Fleet Diagnostics", True),
    ("Deep Scan", "Deep Scan", True),
    ("Explorer Relevance", "Why This Matters", True),
]
_loaded = data_is_loaded()
with st.sidebar:
    st.markdown('<div class="navtitle">◎ NAVIGATION</div>', unsafe_allow_html=True)
    for _label, _prefix, _needs in NAV_ITEMS:
        if _needs and not _loaded:
            # Looks locked; clicking bounces the user to the uplink panel.
            if st.button("🔒  " + _label, key="navlock_" + _prefix, use_container_width=True,
                         help="Load test data first — this link takes you to the uplink."):
                st.session_state["_scroll_to"] = "Telemetry Uplink"
        else:
            if st.button("▸  " + _label, key="nav_" + _prefix, use_container_width=True):
                st.session_state["_scroll_to"] = _prefix
    if not _loaded:
        st.markdown('<div class="navlocked">🔒 Upload or select test data first '
                    'to unlock the diagnostics sections.</div>', unsafe_allow_html=True)
    st.markdown('<div class="navfoot">Select a section to jump to it.</div>',
                unsafe_allow_html=True)

if st.session_state.get("_scroll_to"):
    components.html(_scroll_js(st.session_state.pop("_scroll_to")), height=0)

# wire the glowing "Run Diagnostics" button to smooth-scroll (delegated, once)
components.html(
    "<script>(function(){var p=window.parent.document,w=window.parent;"
    "if(w.__kmcJump)return; w.__kmcJump=true;"
    "function go(x,n){var els=[].slice.call(p.querySelectorAll('.sect'));"
    "var t=els.filter(function(e){return e.textContent.trim().toLowerCase().indexOf(x.toLowerCase())===0;})[0];"
    "if(t){t.scrollIntoView({behavior:'smooth',block:'start'});}else if(n>0){setTimeout(function(){go(x,n-1);},120);}}"
    "p.addEventListener('click',function(ev){var j=ev.target.closest&&ev.target.closest('.jump-btn');"
    "if(j){ev.preventDefault();go('Fleet Diagnostics',18);}});})();</script>", height=0)

# ---- one-time sci-fi boot splash (fades out automatically) ----
if "booted" not in st.session_state:
    st.session_state.booted = True
    st.markdown("""
<style>
#boot{ position:fixed; inset:0; z-index:99999; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:20px; font-family:'Share Tech Mono',monospace;
  background:#05060f; animation:bootfade .7s ease 2.7s forwards; }
@keyframes bootfade{ to{ opacity:0; visibility:hidden; } }
#boot h2{ font-family:'Orbitron',sans-serif; font-weight:900; letter-spacing:5px; margin:0;
  color:#eaf6ff; text-shadow:0 0 20px rgba(34,211,238,.6), 0 0 44px rgba(34,211,238,.3); }
#boot .log{ min-width:300px; min-height:118px; color:#67e8f9; font-size:.84rem; letter-spacing:1px; }
#boot .log div{ opacity:0; margin:4px 0; animation:bln .4s ease forwards; }
#boot .log div:nth-child(1){ animation-delay:.25s }
#boot .log div:nth-child(2){ animation-delay:.70s }
#boot .log div:nth-child(3){ animation-delay:1.15s }
#boot .log div:nth-child(4){ animation-delay:1.60s }
#boot .log div:nth-child(5){ color:#34d399; animation-delay:2.15s }
@keyframes bln{ from{ opacity:0; transform:translateY(5px) } to{ opacity:1; transform:none } }
#boot .bar{ width:280px; height:6px; border-radius:999px; background:rgba(34,211,238,.15); overflow:hidden; }
#boot .bar i{ display:block; height:100%; width:0; border-radius:999px;
  background:linear-gradient(90deg,#22d3ee,#e879f9); box-shadow:0 0 14px #22d3ee;
  animation:bfill 2.4s cubic-bezier(.6,.05,.3,1) forwards; }
@keyframes bfill{ to{ width:100% } }
</style>
<div id="boot">
  <h2>KEPLER MISSION CONTROL</h2>
  <div class="log">
    <div>&gt; INITIALIZING SYSTEMS<span>_</span></div>
    <div>&gt; CALIBRATING DEEP-SPACE SENSORS...</div>
    <div>&gt; LINKING TELEMETRY ARRAY...</div>
    <div>&gt; LOADING CLASSIFIER FLEET [5]...</div>
    <div>&gt; ALL SYSTEMS ONLINE &#9608;</div>
  </div>
  <div class="bar"><i></i></div>
</div>
""", unsafe_allow_html=True)

# ---- hero header ----
st.markdown("""
<div class="hero">
  <div class="sys">
    <div class="ring"></div><div class="planet"></div><div class="moon"></div>
  </div>
  <div>
    <h1>KEPLER MISSION CONTROL</h1>
    <div class="sub">EXOPLANET DISPOSITION CLASSIFIER</div>
    <div class="tag">Five supervised classifiers evaluated on labelled Kepler Object of Interest
    telemetry. Upload a held-out test set to run live diagnostics — models are pre-trained and
    never re-fitted in flight.</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not METADATA_PATH.exists():
    st.error("Model artifacts are missing. Run train_models.py first and commit the generated model/ files.")
    st.stop()

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]
model_files = metadata["models"]
classes_meta = metadata.get("classes", [])

# ---- HUD status strip ----
st.markdown(f"""
<div class="hud">
  <div class="chip live">SYSTEMS ONLINE</div>
  <div class="chip">MODELS&nbsp;LOADED&nbsp;<b>{len(model_files)}</b></div>
  <div class="chip">INPUT&nbsp;FEATURES&nbsp;<b>{len(feature_columns)}</b></div>
  <div class="chip">TARGET&nbsp;CLASSES&nbsp;<b>{len(classes_meta)}</b></div>
  <div class="chip">TARGET&nbsp;//&nbsp;<b>{target_column}</b></div>
</div>
""", unsafe_allow_html=True)

# ---- telemetry uplink (upload or sample) — kept at the top for quick access ----
st.markdown('<span class="anchor" id="uplink"></span>', unsafe_allow_html=True)
st.markdown('<div class="holo">', unsafe_allow_html=True)
st.markdown('<div class="sect">Telemetry Uplink</div>', unsafe_allow_html=True)
st.caption("Evaluate the saved pipelines on labelled test data — upload your own, or run the "
           "bundled sample. No retraining happens here.")

options = [OPT_UPLOAD] + ([OPT_SAMPLE] if SAMPLE_PATH.exists() else [])
source_choice = st.radio("Data source", options, horizontal=True,
                         label_visibility="collapsed", key="src_choice")

uploaded_file = None
data_source = None     # a file-like (upload) or a Path (sample)
data_label = None
if source_choice == OPT_UPLOAD:
    uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"],
                                     key="uploader")
    if uploaded_file is not None:
        data_source, data_label = uploaded_file, uploaded_file.name
    else:
        st.caption(f"◆ Need {len(feature_columns)} feature columns + target `{target_column}`. "
                   f"Scroll down for the briefing — or switch to the bundled sample above.")
else:
    data_source, data_label = SAMPLE_PATH, "test_data.csv (bundled Kepler test set)"
    st.success(f"🛰 Using bundled sample — **{SAMPLE_PATH.name}**, the held-out test split.")

if SAMPLE_PATH.exists():
    st.download_button("⬇ Download sample test_data.csv", SAMPLE_PATH.read_bytes(),
                       file_name="test_data.csv", mime="text/csv",
                       help="Grab the expected CSV format to inspect or re-upload.")
if data_source is not None:
    st.markdown('<a href="#diagnostics" class="jump-btn">▼ RUN DIAGNOSTICS ▼</a>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- mission objective (aim of the test) ----
st.markdown('<span class="anchor" id="objective"></span>', unsafe_allow_html=True)
st.markdown('<div class="holo">', unsafe_allow_html=True)
st.markdown('<div class="sect">Mission Objective — Aim of This Test</div>', unsafe_allow_html=True)
st.markdown(
    f"**Aim:** determine which machine-learning classifier most reliably sorts Kepler Objects "
    f"of Interest into their true disposition — **CONFIRMED planet**, **CANDIDATE**, or "
    f"**FALSE POSITIVE** — using only measured transit and stellar signals.\n\n"
    f"**How the test works:**\n"
    f"- Five pre-trained classifiers — Logistic Regression, Decision Tree, KNN, "
    f"Gaussian Naive Bayes and SVM — are run on the **same** uploaded, labelled test set "
    f"(`{target_column}`, {len(feature_columns)} input features). No model is retrained here.\n"
    f"- Each is scored on six metrics — **Accuracy, AUC, Precision, Recall, F1, MCC** — so "
    f"strong *and* weak performers are exposed, including on the imbalanced minority classes.\n"
    f"- The winner is the model that best balances catching real planets against rejecting "
    f"false alarms — the classifier an observatory would trust to **triage its follow-up queue**.",
    unsafe_allow_html=False,
)
st.markdown('</div>', unsafe_allow_html=True)

# ---- KOI field guide (domain primer) ----
st.markdown('<span class="anchor" id="koi"></span>', unsafe_allow_html=True)
st.markdown('<div class="holo">', unsafe_allow_html=True)
st.markdown('<div class="sect">KOI Field Guide — Kepler Objects of Interest</div>', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Rajdhani',sans-serif; color:#dbeafe; font-size:.98rem; line-height:1.5;">
A <b>Kepler Object of Interest (KOI)</b> is a star the Kepler telescope flagged for a
periodic dip in brightness — a possible <b>transit</b>. When a planet crosses in front of
its star it blocks a sliver of light; the <b>depth</b>, <b>duration</b> and <b>period</b> of
those dips encode the planet's size and orbit. Each KOI is given a <i>disposition</i>:
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:14px 0 6px;">
  <div style="border-left:3px solid #34d399; border-radius:10px; padding:12px 14px;
       background:rgba(52,211,153,.08);">
    <div style="font-family:'Orbitron'; color:#34d399; letter-spacing:1px;">CONFIRMED</div>
    <div style="color:#cfe3ff; font-size:.88rem; margin-top:5px;">Validated as a real
      exoplanet through follow-up observation or statistical validation.</div>
  </div>
  <div style="border-left:3px solid #fbbf24; border-radius:10px; padding:12px 14px;
       background:rgba(251,191,36,.08);">
    <div style="font-family:'Orbitron'; color:#fbbf24; letter-spacing:1px;">CANDIDATE</div>
    <div style="color:#cfe3ff; font-size:.88rem; margin-top:5px;">Passes automated vetting
      and looks planet-like, but is not yet confirmed — needs more data.</div>
  </div>
  <div style="border-left:3px solid #e879f9; border-radius:10px; padding:12px 14px;
       background:rgba(232,121,249,.08);">
    <div style="font-family:'Orbitron'; color:#e879f9; letter-spacing:1px;">FALSE POSITIVE</div>
    <div style="color:#cfe3ff; font-size:.88rem; margin-top:5px;">The dip is an impostor —
      eclipsing binary stars, background blends, or instrument / stellar noise.</div>
  </div>
</div>
""", unsafe_allow_html=True)
with st.expander("▸ Key telemetry the models read"):
    st.markdown("""
- **`koi_period`** — orbital period (days) between transits
- **`koi_duration`** — how long each transit lasts (hours)
- **`koi_depth`** — fractional brightness drop during transit (ppm)
- **`koi_prad`** — inferred planet radius (Earth radii)
- **`koi_teq` / `koi_insol`** — equilibrium temperature & stellar flux received
- **`koi_model_snr`** — signal-to-noise of the transit detection
- **`koi_steff` / `koi_slogg` / `koi_srad`** — host-star temperature, surface gravity, radius
- **`koi_kepmag`** — apparent brightness of the star

*Vetting flags and the pipeline's own score are **excluded** from training — they would
leak the answer, so the models must judge from physics alone.*
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---- dataset dossier (data dictionary + rationale) ----
_PILL = {
    "USED": "color:#05060f;background:#34d399;",
    "EXCLUDED": "color:#fbdce9;background:rgba(232,121,249,.22);border:1px solid #e879f9;",
    "TARGET": "color:#05060f;background:#22d3ee;",
}
_DOSSIER = [
    ("Identifiers", "<code>kepid</code>, <code>kepoi_name</code>, <code>kepler_name</code>",
     "Catalogue IDs for the star, the KOI, and the named planet.", "EXCLUDED",
     "Not physical signals — and <code>kepler_name</code> exists only for CONFIRMED planets, so it would hand the model the answer."),
    ("Pipeline verdict", "<code>koi_pdisposition</code>, <code>koi_score</code>",
     "The Kepler pipeline's own call and its 0–1 confidence.", "EXCLUDED",
     "Direct target leakage — these already encode the label we are trying to predict."),
    ("Vetting flags", "<code>koi_fpflag_nt / ss / co / ec</code>",
     "Automated reasons a signal looks like a false positive.", "EXCLUDED",
     "They essentially <i>define</i> the FALSE POSITIVE class; training on them is leakage."),
    ("Orbital timing &amp; geometry", "<code>koi_period</code>, <code>koi_time0bk</code>, <code>koi_impact</code>, <code>koi_duration</code>",
     "Orbital period, transit epoch, crossing geometry and duration.", "USED",
     "Real planets show stable, physically consistent timing; many impostors do not."),
    ("Transit signal", "<code>koi_depth</code>, <code>koi_model_snr</code>",
     "Brightness drop during transit and detection signal-to-noise.", "USED",
     "Depth scales with planet size; low SNR flags marginal or spurious detections."),
    ("Inferred planet", "<code>koi_prad</code>, <code>koi_teq</code>, <code>koi_insol</code>",
     "Planet radius, equilibrium temperature and stellar flux received.", "USED",
     "Implausibly large radii and extreme values separate planets from eclipsing binaries."),
    ("Host star", "<code>koi_steff</code>, <code>koi_slogg</code>, <code>koi_srad</code>, <code>koi_kepmag</code>",
     "Star temperature, surface gravity, radius and brightness.", "USED",
     "The star sets the context that turns a transit into a planet size — needed to judge plausibility."),
    ("Measurement errors", "<code>*_err1</code>, <code>*_err2</code>",
     "Uncertainty (error bars) on each measurement above.", "USED",
     "Large uncertainties correlate with unreliable, often false-positive, detections — useful signal."),
    ("Sky position / batch", "<code>ra_str</code>, <code>dec_str</code>, <code>koi_tce_delivname</code>",
     "Sky coordinates and processing-batch name, stored as text.", "EXCLUDED",
     "Non-numeric and not physically predictive of disposition."),
    ("Target label", "<code>koi_disposition</code>",
     "The verified outcome: CONFIRMED / CANDIDATE / FALSE POSITIVE.", "TARGET",
     "This is what every model learns to predict."),
]
_rows = "".join(
    f'<tr>'
    f'<td style="padding:9px 10px;border-bottom:1px solid rgba(103,232,249,.12);vertical-align:top;">'
    f'<b style="color:#eaf6ff;">{g}</b><br><span style="color:#8aa0c8;font-size:.82rem;">{cols}</span></td>'
    f'<td style="padding:9px 10px;border-bottom:1px solid rgba(103,232,249,.12);color:#cfe3ff;font-size:.86rem;vertical-align:top;">{tells}</td>'
    f'<td style="padding:9px 10px;border-bottom:1px solid rgba(103,232,249,.12);vertical-align:top;white-space:nowrap;">'
    f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.68rem;padding:3px 9px;border-radius:6px;letter-spacing:1px;{_PILL[flag]}">{flag}</span></td>'
    f'<td style="padding:9px 10px;border-bottom:1px solid rgba(103,232,249,.12);color:#cfe3ff;font-size:.86rem;vertical-align:top;">{why}</td>'
    f'</tr>'
    for g, cols, tells, flag, why in _DOSSIER
)
st.markdown('<span class="anchor" id="dossier"></span>', unsafe_allow_html=True)
st.markdown('<div class="holo">', unsafe_allow_html=True)
st.markdown('<div class="sect">Dataset Dossier — What the Telemetry Contains</div>', unsafe_allow_html=True)
st.markdown(
    f"The raw KOI catalogue carries ~50 columns. The models keep only the **physical "
    f"measurements** ({len(feature_columns)} numeric features) and learn to predict "
    f"`{target_column}` from them — everything that would leak the answer is dropped:",
    unsafe_allow_html=False)
st.markdown(
    '<div style="overflow-x:auto;">'
    '<table style="width:100%;min-width:560px;border-collapse:collapse;font-family:\'Rajdhani\',sans-serif;">'
    '<tr style="text-align:left;color:#8aa0c8;font-family:\'Share Tech Mono\',monospace;font-size:.72rem;letter-spacing:1px;text-transform:uppercase;">'
    '<th style="padding:6px 10px;">Signal group</th><th style="padding:6px 10px;">What it tells us</th>'
    '<th style="padding:6px 10px;">In model?</th><th style="padding:6px 10px;">Why</th></tr>'
    f'{_rows}</table></div>', unsafe_allow_html=True)
st.caption("Leakage principle: a feature is excluded if it already knows the verdict "
           "(pipeline score, FP flags) or merely tags the row (IDs) — the model must earn its "
           "classification from physics, not from the catalogue's own bookkeeping.")
st.markdown('</div>', unsafe_allow_html=True)

if data_source is None:
    st.info(f"Awaiting uplink — upload a CSV (or pick the bundled sample above). It must "
            f"contain {len(feature_columns)} input features and target column `{target_column}`.")
    with st.expander("▸ Required columns"):
        st.write(feature_columns + [target_column])
    st.stop()

st.caption(f"▸ Evaluating: **{data_label}**")
try:
    test_df = pd.read_csv(data_source)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

required_columns = feature_columns + [target_column]
missing = [c for c in required_columns if c not in test_df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if test_df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_test = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()

if y_test.isna().any():
    st.error(f"Target column `{target_column}` contains missing values. Please provide labelled test rows.")
    st.stop()


def calculate_auc(model, X, y):
    if not hasattr(model, "predict_proba"):
        return float("nan")
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if len(classes) == 2:
        return roc_auc_score(y, probabilities[:, 1])
    return roc_auc_score(
        y, probabilities, labels=classes, multi_class="ovr", average="weighted"
    )


def evaluate_one(name, filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing saved model: {path.name}")
    pipeline = joblib.load(path)
    predictions = pipeline.predict(X_test)
    try:
        auc = calculate_auc(pipeline, X_test, y_test)
    except ValueError as exc:
        auc = float("nan")
        st.warning(f"AUC for {name} could not be calculated: {exc}")
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
        "pipeline": pipeline,
        "predictions": predictions,
    }


METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]


def _cell(value):
    if pd.isna(value):
        return '<td>—</td>'
    alpha = max(0.0, min(1.0, float(value)))  # MCC can be <0; clamp for the glow
    bg = f"background:rgba(34,211,238,{0.08 + alpha * 0.30});"
    return f'<td style="{bg}">{value:.4f}</td>'


def comparison_table_html(rows, winner):
    head = "".join(f"<th>{k}</th>" for k in METRIC_KEYS)
    body = ""
    for r in rows:
        cls = ' class="win"' if r["Model"] == winner else ""
        badge = '<span class="win-badge">TOP</span>' if r["Model"] == winner else ""
        cells = "".join(_cell(r[k]) for k in METRIC_KEYS)
        body += f'<tr{cls}><td>{r["Model"]}{badge}</td>{cells}</tr>'
    return (f'<div style="overflow-x:auto;"><table class="holo-table" style="min-width:560px;">'
            f'<tr><th>Classifier</th>{head}</tr>{body}</table></div>')


def tiles_html(result):
    tiles = ""
    for k in METRIC_KEYS:
        v = result[k]
        txt = "N/A" if pd.isna(v) else f"{v:.4f}"
        tiles += f'<div class="tile"><div class="k">{k}</div><div class="v">{txt}</div></div>'
    return f'<div class="tiles">{tiles}</div>'


def _style_ax(fig, ax, ylim=(0, 1)):
    """Apply the holographic dark theme to a matplotlib axes."""
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(colors="#dbeafe", labelsize=8)
    ax.yaxis.grid(True, color="#22d3ee", alpha=0.12, linewidth=0.8)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color("#22d3ee")
        spine.set_alpha(0.35)


def fleet_chart(rows):
    """Grouped bars: each model vs. the four headline metrics."""
    metrics = ["Accuracy", "AUC", "F1 Score", "MCC"]
    colors = ["#22d3ee", "#e879f9", "#34d399", "#fbbf24"]
    names = [r["Model"] for r in rows]
    short = [n.replace(" ", "\n") for n in names]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [0.0 if pd.isna(r[m]) else float(r[m]) for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=c,
               edgecolor="#05060f", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8, color="#dbeafe")
    ax.set_ylabel("Score", color="#67e8f9", fontsize=9)
    leg = ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
                    frameon=False, fontsize=8, labelcolor="#dbeafe")
    _style_ax(fig, ax)
    return fig


def perclass_chart(report, labels):
    """Grouped bars: precision / recall / F1 for each target class."""
    metrics = ["precision", "recall", "f1-score"]
    colors = ["#22d3ee", "#e879f9", "#fbbf24"]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [report.get(str(l), {}).get(m, 0.0) for l in labels]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m.title(), color=c,
                      edgecolor="#05060f", linewidth=0.6)
        ax.bar_label(bars, fmt="%.2f", color="#dbeafe", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels], fontsize=8, color="#dbeafe")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              frameon=False, fontsize=8, labelcolor="#dbeafe")
    _style_ax(fig, ax, ylim=(0, 1.08))
    return fig


def _find_class(labels, *keywords):
    """Locate a class label by keyword (robust to other datasets)."""
    for l in labels:
        low = str(l).lower()
        if any(k in low for k in keywords):
            return l
    return None


def mission_debrief(name, result, cm, labels):
    """Plain-language interpretation + space-explorer relevance, from the matrix."""
    labels = list(labels)
    idx = {l: i for i, l in enumerate(labels)}

    def recall_of(lbl):
        i = idx[lbl]
        tot = cm[i, :].sum()
        return float(cm[i, i] / tot) if tot else float("nan")

    def precision_of(lbl):
        i = idx[lbl]
        tot = cm[:, i].sum()
        return float(cm[i, i] / tot) if tot else float("nan")

    planet = _find_class(labels, "confirm")       # real planets
    junk = _find_class(labels, "false")           # false positives
    acc, auc, f1, mcc = result["Accuracy"], result["AUC"], result["F1 Score"], result["MCC"]

    lines = [
        f"**{name}** scores **{acc:.1%} accuracy**, "
        f"**AUC {auc:.3f}**, **F1 {f1:.3f}** and **MCC {mcc:.3f}** on the held-out test set. "
        f"AUC above ~0.85 means the model ranks a random real detection above a random "
        f"non-detection most of the time — strong separating power; MCC rewards getting "
        f"*all* classes right, not just the majority."
    ]
    if planet is not None:
        pr, pp = recall_of(planet), precision_of(planet)
        lines.append(
            f"**Finding real planets →** it recovers **{pr:.1%}** of true `{planet}` "
            f"signals (recall), and when it calls something `{planet}` it is right "
            f"**{pp:.1%}** of the time (precision). The remaining {1-pr:.1%} are real "
            f"planets it would miss on a first pass."
        )
    if junk is not None:
        jr = recall_of(junk)
        lines.append(
            f"**Rejecting noise →** it correctly screens out **{jr:.1%}** of `{junk}` "
            f"signals — eclipsing binaries, instrument artefacts and other look-alikes "
            f"that waste telescope time."
        )
    return "\n\n".join(lines)


results = []
failures = []
with st.spinner("◢ SCANNING TELEMETRY · LOADING CLASSIFIER FLEET · COMPUTING DIAGNOSTICS..."):
    for name, filename in model_files.items():
        try:
            results.append(evaluate_one(name, filename))
        except Exception as exc:
            failures.append(f"{name}: {exc}")

if failures:
    st.error("Some models could not be evaluated:\n\n" + "\n".join(failures))
if not results:
    st.stop()

# Winner by F1 (ties broken by AUC), matching the README methodology.
winner = max(results, key=lambda r: (r["F1 Score"], 0 if pd.isna(r["AUC"]) else r["AUC"]))["Model"]

# ---- comparison panel ----
st.markdown('<span class="anchor" id="diagnostics"></span>', unsafe_allow_html=True)
st.markdown('<div class="holo">', unsafe_allow_html=True)
st.markdown('<div class="sect">Fleet Diagnostics — Performance Matrix</div>', unsafe_allow_html=True)
st.markdown(comparison_table_html(results, winner), unsafe_allow_html=True)
fig_fleet = fleet_chart(results)
st.pyplot(fig_fleet)
plt.close(fig_fleet)
win_row = next(r for r in results if r["Model"] == winner)
st.success(
    f"🏆 **Fleet winner: {winner}** — F1 {win_row['F1 Score']:.3f}, "
    f"AUC {win_row['AUC']:.3f}, accuracy {win_row['Accuracy']:.1%}. "
    "It leads the squadron on overall detection quality."
)
st.markdown('</div>', unsafe_allow_html=True)

# ---- model selection ----
model_names = [r["Model"] for r in results]
default_idx = model_names.index(winner) if winner in model_names else 0
st.markdown('<div class="scan-label">◎ Deep-scan target &nbsp;//&nbsp; '
            '<b>select a classifier to inspect</b></div>', unsafe_allow_html=True)
selected_model = st.selectbox("Select classifier for deep scan", model_names,
                              index=default_idx, label_visibility="collapsed")

selected_result = next((r for r in results if r["Model"] == selected_model), None)
if selected_result is not None:
    st.markdown('<span class="anchor" id="deepscan"></span>', unsafe_allow_html=True)
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.markdown(f'<div class="sect">Deep Scan // {selected_model}</div>', unsafe_allow_html=True)
    st.markdown(tiles_html(selected_result), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="holo">', unsafe_allow_html=True)
        st.markdown('<div class="sect">Confusion Matrix</div>', unsafe_allow_html=True)
        labels = selected_result["pipeline"].classes_
        cm = confusion_matrix(y_test, selected_result["predictions"], labels=labels)

        holo_cmap = LinearSegmentedColormap.from_list(
            "holo", ["#060b1e", "#0e7490", "#22d3ee", "#a5f3fc"])
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        sns.heatmap(cm, annot=True, fmt="d", cmap=holo_cmap, cbar=False,
                    linewidths=1.2, linecolor="#05060f",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"color": "#eaf6ff", "fontsize": 12, "weight": "bold"})
        ax.set_xlabel("PREDICTED", color="#67e8f9", labelpad=10, fontsize=10)
        ax.set_ylabel("ACTUAL", color="#67e8f9", labelpad=10, fontsize=10)
        ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values():
            spine.set_color("#22d3ee")
            spine.set_alpha(0.4)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="holo">', unsafe_allow_html=True)
        st.markdown('<div class="sect">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(
            y_test, selected_result["predictions"], zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- per-class detection + mission debrief ----
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown('<div class="holo">', unsafe_allow_html=True)
        st.markdown('<div class="sect">Per-Class Detection</div>', unsafe_allow_html=True)
        st.caption("How reliably this model handles each disposition — the metric that "
                   "decides whether a real planet is caught or a false alarm slips through.")
        fig_pc = perclass_chart(report, labels)
        st.pyplot(fig_pc)
        plt.close(fig_pc)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_d:
        st.markdown('<div class="holo">', unsafe_allow_html=True)
        st.markdown('<div class="sect">Mission Debrief</div>', unsafe_allow_html=True)
        st.markdown(mission_debrief(selected_model, selected_result, cm, labels))
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- why it matters to explorers ----
    st.markdown('<span class="anchor" id="explorer"></span>', unsafe_allow_html=True)
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.markdown('<div class="sect">Why This Matters to Space Explorers</div>', unsafe_allow_html=True)
    st.markdown(
        "Kepler stared at ~150,000 stars and flagged **thousands** of Objects of Interest — "
        "far more than astronomers can confirm by hand. Each real confirmation needs scarce, "
        "expensive follow-up: additional transits, radial-velocity measurements, high-resolution "
        "imaging. A classifier like this acts as an **automated first-pass vetter**, exactly how "
        "the mission's own *Robovetter* and Google's *AstroNet* (which helped discover Kepler-90i) "
        "were used:\n\n"
        "- **Triage** — rank new telemetry so the most planet-like signals reach the top of the "
        "observation queue.\n"
        "- **Filter noise** — auto-reject eclipsing binaries and instrument artefacts, saving "
        "telescope hours for genuine candidates.\n"
        "- **Prioritise under uncertainty** — the `CANDIDATE` class marks signals needing a human "
        "or more data; the model routes effort there instead of to obvious false positives.\n\n"
        "**Reality check:** this is a *decision aid*, not a discovery pipeline. It is trained only "
        "on measured transit/stellar features (leakage columns removed), so a flagged planet still "
        "requires real astronomical confirmation — the model buys *speed and prioritisation*, not "
        "proof."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.markdown('<div class="sect">Actual vs Predicted — Signal Log</div>', unsafe_allow_html=True)
    predictions_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": pd.Series(selected_result["predictions"])
    })
    predictions_df["Match"] = np.where(
        predictions_df["Actual"] == predictions_df["Predicted"], "✔", "✘")
    st.dataframe(predictions_df.head(100), use_container_width=True)
    st.download_button(
        "⬇ Download predictions CSV",
        predictions_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)
