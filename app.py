"""Incident Lens — learn from offshore incident reports without reading every one.

Run: uv run streamlit run app.py
"""
from pathlib import Path
import json, pickle, re, html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

_ax = dict(gridcolor="rgba(94,234,212,.08)", zeroline=False, linecolor="rgba(94,234,212,.18)", showline=False,
           tickfont=dict(color="#8FB3BF", size=11), title_font=dict(color="#8FB3BF", size=12))
pio.templates["ocean"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", color="#D5EEEA", size=12),
    title=dict(font=dict(family="Inter, system-ui, sans-serif", size=15, color="#E6F4F1"), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=_ax, yaxis=_ax,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#BFE3DD", size=11)),
    hoverlabel=dict(bgcolor="#0B2233", bordercolor="#2DD4BF", font=dict(family="Inter", color="#E6F4F1")),
    barcornerradius=5, bargap=0.28,
    colorway=["#2DD4BF", "#38BDF8", "#A78BFA", "#FBBF24", "#FF7A45", "#F472B6"]))
pio.templates.default = "ocean"
PCFG = {"displayModeBar": False}
import streamlit.components.v1 as components
from galaxy_component import galaxy_html

ART = Path(__file__).parent / "artifacts"
HERE_LAB = Path(__file__).parent / "labeling"
CODE2EL = {"HU": "3", "EQ": "15", "MS": "8", "SU": "17", "CO": "9", "WE": "6"}
COLORS = {"Human performance": "#2DD4BF", "Equipment failure": "#38BDF8", "Management systems": "#A78BFA",
          "Supervision": "#F472B6", "Work environment": "#FBBF24", "Communication": "#FF7A45"}
AC, HOT, GREY = "#2DD4BF", "#FF7A45", "#3E5A6B"

st.set_page_config(page_title="Incident Lens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&family=Instrument+Serif:ital@0;1&display=swap');
:root{--bg:#061826;--sf:#0B2233;--sf2:#0E2A3F;--ln:rgba(94,234,212,.12);--tx:#E6F4F1;--mu:#7FA7B5;--ac:#2DD4BF;--ac2:#5EEAD4;--hot:#FF7A45;}
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
.stApp {background: radial-gradient(1200px 600px at 85% -10%, rgba(45,212,191,.10), transparent 60%),
  radial-gradient(900px 500px at -10% 30%, rgba(56,189,248,.07), transparent 60%), var(--bg);}
.block-container {padding-top: 1.4rem; max-width: 1280px;}
h1,h2,h3,h4 {letter-spacing:-.02em; color: var(--tx);}
.hero {position:relative; padding: 30px 34px 26px; border-radius: 22px; margin-bottom: 18px; overflow:hidden;
  background: linear-gradient(135deg, rgba(14,42,63,.9), rgba(6,24,38,.9)); border: 1px solid var(--ln);}
.hero:before {content:""; position:absolute; inset:-40% -10% auto auto; width:520px; height:520px; border-radius:50%;
  background: radial-gradient(circle, rgba(45,212,191,.22), transparent 65%); filter: blur(10px);}
.eyebrow {font: 600 .74rem 'JetBrains Mono', monospace; letter-spacing:.18em; color: var(--ac2);}
.hero h1 {margin: 8px 0 0; font-size: 2.6rem; font-weight: 800; line-height:1.08; position:relative;}
.hero h1 em {font-style:normal; background: linear-gradient(90deg,#5EEAD4,#38BDF8); -webkit-background-clip:text; color:transparent;}
.hero p {margin: 10px 0 0; color: var(--mu); font-size: 1.04rem; max-width: 760px; position:relative;}
.kpis {display:flex; gap: 34px; margin-top: 22px; position:relative; flex-wrap:wrap;}
.kpis .v {font-size: 2rem; font-weight: 800; color: var(--tx); line-height:1; font-variant-numeric: tabular-nums;}
.kpis .l {font-size: .78rem; color: var(--mu); margin-top: 6px; text-transform: uppercase; letter-spacing:.08em;}
@property --n {syntax:'<integer>'; initial-value:0; inherits:false;}
.count {animation: cnt 1.8s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n var(--n);}
.count:after {content: counter(n);}
@keyframes cnt {to {--n: var(--t);}}
.chip {display:inline-block; padding: 4px 11px; border-radius: 999px; font-size: .76rem; margin: 0 6px 6px 0;
  background: rgba(45,212,191,.07); border: 1px solid var(--ln); color:#BFE3DD;}
.card {padding: 16px 18px; border-radius: 16px; background: linear-gradient(180deg, var(--sf2), var(--sf)); border: 1px solid var(--ln);
  margin-bottom: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.25);}
.card h4 {margin: 0 0 6px; font-size: 1.02rem;}
.sim {float:right; font: 600 .82rem 'JetBrains Mono', monospace; color: var(--bg); background: var(--ac); padding: 2px 9px; border-radius: 999px;}
.muted {color: var(--mu); font-size:.88rem;}
.lesson {border-left: 3px solid var(--ac); padding: 7px 12px; margin: 7px 0; background: rgba(45,212,191,.07); border-radius: 0 10px 10px 0;}
.cause {border-left: 3px solid #38BDF8; padding: 5px 11px; margin: 6px 0; background: rgba(56,189,248,.06); border-radius: 0 10px 10px 0;}
.big {font-size: 2.4rem; font-weight: 800; line-height: 1;}
.kpi {padding: 18px; border-radius: 16px; background: linear-gradient(180deg, var(--sf2), var(--sf)); border:1px solid var(--ln); text-align:center;}
.kpi .lbl {color: var(--mu); font-size:.8rem; margin-top:8px;}
.w {padding: 1px 3px; border-radius: 5px;}
.sec {font: 600 .72rem 'JetBrains Mono', monospace; letter-spacing:.16em; color: var(--ac2); margin: 4px 0 2px;}
@keyframes rise {from {opacity:0; transform: translateY(10px);} to {opacity:1; transform:none;}}

.card:hover {border-color: rgba(45,212,191,.45); transform: translateY(-2px); transition: all .2s;}
[role="tablist"] {gap: 6px; background: rgba(8,44,68,.65); padding: 6px; border-radius: 14px; border:1px solid var(--ln); backdrop-filter: blur(6px);}
[role="tab"] {font-size: .95rem; padding: 8px 16px !important; border-radius: 10px; height:auto; border-bottom:0 !important;}
[role="tab"][aria-selected="true"] {background: rgba(45,212,191,.16) !important; color: var(--ac2) !important;}
[role="tab"][aria-selected="true"] p {color: var(--ac2) !important;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .stTabs [data-testid="stTabsHighlight"] {display:none !important;}
.stButton button[kind="primary"] {background: linear-gradient(90deg,#FF7A45,#F97316); border:0; color:#fff; font-weight:700;}
textarea {border-radius: 12px !important;}

.hero:after {content:""; position:absolute; right:-60px; top:-30px; width:620px; height:400px; pointer-events:none;
  background-size: cover; background-position:center; opacity:.55; mix-blend-mode: screen;
  -webkit-mask-image: radial-gradient(closest-side, #000 55%, transparent 100%); mask-image: radial-gradient(closest-side, #000 55%, transparent 100%);
  animation: drift 60s ease-in-out infinite alternate;}
@keyframes drift {from {transform: rotate(-4deg) scale(1);} to {transform: rotate(4deg) scale(1.08);}}
.hero > * {position:relative; z-index:1;}
@supports (animation-timeline: view()) {
  .lesson, .cause, [data-testid="stPlotlyChart"], .bars, .statbig {
    animation: enter linear both; animation-timeline: view(); animation-range: entry 0% entry 35%;}
  @keyframes enter {from {opacity:0; transform: translateY(26px) scale(.98);} to {opacity:1; transform:none;}}
}
.ring {float:right; width:58px; height:58px; border-radius:50%; margin:-4px 0 6px 12px; display:grid; place-items:center;
  background: conic-gradient(#2DD4BF calc(var(--p) * 1%), rgba(94,234,212,.10) 0); box-shadow: 0 0 18px rgba(45,212,191,.25);}
.ring span {width:46px; height:46px; border-radius:50%; background:#0C2E44; display:grid; place-items:center;
  font: 700 .82rem 'JetBrains Mono', monospace; color:#CCFBF1; line-height:1;}
.ring span small {display:block; font: 500 .52rem Inter, sans-serif; color:#7FA7B5; letter-spacing:.08em;}
.bars {padding: 14px 16px; border-radius:16px; background: linear-gradient(180deg, rgba(16,58,82,.6), rgba(10,40,62,.5)); border:1px solid rgba(94,234,212,.10);}
.bar {display:grid; grid-template-columns: 148px 1fr 46px; gap:10px; align-items:center; margin:9px 0; font-size:.86rem; color:#BFE3DD;}
.bar .tr {height:14px; border-radius:99px; background: rgba(94,234,212,.07); overflow:hidden;}
.bar .fl {height:100%; border-radius:99px; width: var(--w); animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both;
  background: linear-gradient(90deg, color-mix(in srgb, var(--c) 55%, transparent), var(--c));
  box-shadow: 0 0 14px color-mix(in srgb, var(--c) 60%, transparent), inset 0 1px 0 rgba(255,255,255,.35);}
.bar b {font: 600 .82rem 'JetBrains Mono', monospace; color:#E6F4F1; text-align:right;}
.bar.top {color:#fff; font-weight:600;}
@keyframes grow {from {width:0;}}
.w {transition: all .2s;}
.mini {display:block; width:100%; height:64px; margin:8px 0 4px;}
.statbig {display:grid; grid-template-columns: auto 1fr; gap: 28px; align-items:center; margin: 10px 0 22px; padding: 22px 28px;
  border-radius: 22px; background: linear-gradient(120deg, rgba(255,122,69,.10), rgba(16,58,82,.55) 45%); border:1px solid rgba(255,122,69,.25);}
.statbig .n {font-family:'Instrument Serif', Georgia, serif; font-size: 7.2rem; line-height:.9; color:#FF7A45;
  text-shadow: 0 0 40px rgba(255,122,69,.35);}
.statbig .t {font-family:'Instrument Serif', Georgia, serif; font-size: 2rem; line-height:1.15; color:#F0FDFA;}
.statbig .t em {color:#5EEAD4;}
.statbig .s {font-size:.92rem; color:#7FA7B5; margin-top:10px; font-family: Inter, sans-serif;}
.lede {font-family:'Instrument Serif', Georgia, serif; font-size: 2.15rem; line-height:1.15; color:#F0FDFA;
  margin: 18px 0 14px; letter-spacing:-.01em; max-width: 980px;}
.lede em {color:#5EEAD4; font-style: italic;}
.kpi {position:relative; overflow:hidden; transition: transform .2s, box-shadow .2s, border-color .2s;}
.kpi:before {content:""; position:absolute; left:18%; right:18%; top:0; height:2px;
  background: linear-gradient(90deg, transparent, #5EEAD4, transparent); box-shadow: 0 0 14px #2DD4BF;}
.kpi:hover {transform: translateY(-3px); border-color: rgba(94,234,212,.45); box-shadow: 0 12px 36px rgba(45,212,191,.18);}
.kpi .big {background: linear-gradient(180deg, #FFFFFF 10%, #9DEFE3 100%); -webkit-background-clip:text; color:transparent;}
::selection {background: rgba(45,212,191,.35); color:#fff;}
::-webkit-scrollbar {width:10px; height:10px;} ::-webkit-scrollbar-track {background: #061826;}
::-webkit-scrollbar-thumb {background: linear-gradient(#14776E, #0E4F5C); border-radius: 10px; border: 2px solid #061826;}
textarea:focus, [data-baseweb="select"]:focus-within, [data-baseweb="input"]:focus-within {box-shadow: 0 0 0 1px #2DD4BF, 0 0 18px rgba(45,212,191,.35) !important;}
[role="tab"] [data-testid="stIconMaterial"] {color: #5EEAD4;}
[data-testid="stPlotlyChart"] {background: linear-gradient(180deg, rgba(14,58,85,.35), rgba(8,40,62,.2)); border:1px solid rgba(94,234,212,.10);
  border-radius: 16px; padding: 10px 12px 4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""<style>
.stApp {background: linear-gradient(180deg, #0F4C68 0%, #0B3A55 22%, #082C44 50%, #061F33 78%, #041626 100%) !important;}
.ocean {position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden;}
.block-container, [data-testid="stMainBlockContainer"] {position:relative; z-index:1;}
.ocean .surf {position:absolute; top:0; left:0; right:0; height:140px;
  background: radial-gradient(60% 120% at 50% -40%, rgba(165,243,252,.28), transparent 70%);}
.ocean .surf:after {content:""; position:absolute; left:-10%; right:-10%; top:0; height:18px; opacity:.35;
  background: radial-gradient(14px 8px at 14px 0, transparent 10px, rgba(204,251,241,.55) 11px, transparent 13px) 0 0/28px 18px repeat-x;
  animation: wave 9s linear infinite;}
@keyframes wave {to {transform: translateX(28px);}}
.ocean b {position:absolute; top:-10%; height:85%; transform-origin: top center; transform: skewX(-12deg);
  background: linear-gradient(180deg, rgba(165,243,252,.16), rgba(165,243,252,0) 85%); filter: blur(14px);
  animation: ray 11s ease-in-out infinite alternate;}
@keyframes ray {0% {opacity:.25; transform: skewX(-16deg) translateX(-30px);} 100% {opacity:.9; transform: skewX(-6deg) translateX(30px);}}
.ocean i {position:absolute; bottom:-30px; border-radius:50%; border:1px solid rgba(204,251,241,.45);
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.55), rgba(165,243,252,.08) 60%);
  animation: rise linear infinite;}
@keyframes rise {0% {transform: translate(0,0); opacity:0;} 10% {opacity:.8;} 50% {transform: translate(14px,-55vh);}
  100% {transform: translate(-8px,-112vh); opacity:0;}}
.ocean .floor {position:absolute; left:0; right:0; bottom:0; height:120px; opacity:.55;}
.gauge {position:fixed; left:26px; top:18vh; height:62vh; width:60px; z-index:0; pointer-events:none;
  border-left:1px solid rgba(94,234,212,.35); font: 500 10px 'JetBrains Mono', monospace; color: rgba(165,243,252,.6);}
.gauge .tk {position:absolute; left:0; width:10px; border-top:1px solid rgba(94,234,212,.5);}
.gauge .tk span {position:absolute; left:14px; top:-7px; white-space:nowrap;}
.gauge .ttl {position:absolute; top:-26px; left:-4px; letter-spacing:.2em; color: rgba(94,234,212,.8);}
.gauge .mk {position:absolute; left:-6px; width:11px; height:11px; border-radius:50%; background:#5EEAD4; top:0;
  box-shadow: 0 0 12px #5EEAD4; animation: dive linear both; animation-timeline: scroll(nearest);}
@keyframes dive {from {top:0;} to {top:100%;}}
.sonar {position:fixed; right:-90px; bottom:-90px; width:300px; height:300px; z-index:0; pointer-events:none;}
.sonar span {position:absolute; inset:0; border-radius:50%; border:1px solid rgba(94,234,212,.45); animation: ping 6s ease-out infinite;}
.sonar span:nth-child(2) {animation-delay:2s;} .sonar span:nth-child(3) {animation-delay:4s;}
.sonar:after {content:""; position:absolute; inset:0; border-radius:50%;
  background: conic-gradient(from 0deg, rgba(94,234,212,.22), transparent 18%); animation: sweep 6s linear infinite;}
@keyframes ping {from {transform: scale(.15); opacity:.9;} to {transform: scale(1); opacity:0;}}
@keyframes sweep {to {transform: rotate(360deg);}}
.hero, .card, .kpi {backdrop-filter: blur(6px);}
.hero {background: linear-gradient(135deg, rgba(16,70,96,.72), rgba(6,31,51,.72)) !important;}
.card, .kpi {background: linear-gradient(180deg, rgba(16,58,82,.78), rgba(10,40,62,.78)) !important;}
@media (max-width: 1400px) { .gauge {display:none;} }

.ocean:before, .ocean:after {content:""; position:absolute; left:-20%; right:-20%; top:0; height:75vh; pointer-events:none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='520' height='520'><filter id='f'><feTurbulence type='turbulence' baseFrequency='.018' numOctaves='1' seed='7'/><feColorMatrix values='0 0 0 0 .80 0 0 0 0 .98 0 0 0 0 .95 -14 0 0 0 1.25'/></filter><rect width='100%' height='100%' filter='url(%23f)'/></svg>"); background-size: 520px 520px; mix-blend-mode: screen; opacity:.10;
  -webkit-mask-image: linear-gradient(180deg,#000 0%,transparent 100%); mask-image: linear-gradient(180deg,#000 0%,transparent 100%);
  animation: caus 38s linear infinite;}
.ocean:after {background-size: 700px 700px; opacity:.06; animation-duration: 52s; animation-direction: reverse; filter: blur(1px);}
@keyframes caus {from {background-position: 0 0;} to {background-position: 520px 260px;}}
[role="tabpanel"] > div {animation: tabin .5s cubic-bezier(.2,.8,.2,1) both;}
@keyframes tabin {from {opacity:0; transform: translateY(14px); filter: blur(4px);} to {opacity:1; transform:none; filter:none;}}
@media (max-width: 760px) {
  .block-container, [data-testid="stMainBlockContainer"] {padding-left: 14px !important; padding-right: 14px !important;}
  .hero {padding: 22px 18px !important;} .hero h1 {font-size: 2.1rem !important; line-height:1.1 !important;}
  .hero:after, .sonar, .gauge {display:none !important;}
  .kpis {gap: 14px 22px !important;}
  .lede {font-size: 1.5rem;}
  .statbig {grid-template-columns: 1fr; gap: 6px; padding: 18px;} .statbig .n {font-size: 4.6rem;} .statbig .t {font-size: 1.4rem;}
  .bar {grid-template-columns: 104px 1fr 40px; font-size: .8rem;}
  [role="tablist"] {overflow-x: auto; flex-wrap: nowrap !important;}
  .ring {width:48px; height:48px;} .ring span {width:38px; height:38px;}
}
</style>
<div class="ocean"><div class="surf"></div><b style='left:8%;animation-delay:-0s;width:120px'></b><b style='left:22%;animation-delay:-3s;width:70px'></b><b style='left:38%;animation-delay:-6s;width:160px'></b><b style='left:55%;animation-delay:-2s;width:90px'></b><b style='left:70%;animation-delay:-8s;width:140px'></b><b style='left:86%;animation-delay:-5s;width:80px'></b><i style='left:33.1%;width:5px;height:5px;animation-duration:24s;animation-delay:-2s'></i><i style='left:53.4%;width:7px;height:7px;animation-duration:15s;animation-delay:-15s'></i><i style='left:5.6%;width:7px;height:7px;animation-duration:15s;animation-delay:-3s'></i><i style='left:42.8%;width:11px;height:11px;animation-duration:16s;animation-delay:-7s'></i><i style='left:62.2%;width:12px;height:12px;animation-duration:23s;animation-delay:-12s'></i><i style='left:95.7%;width:4px;height:4px;animation-duration:28s;animation-delay:-9s'></i><i style='left:15.8%;width:5px;height:5px;animation-duration:19s;animation-delay:-24s'></i><i style='left:19.3%;width:9px;height:9px;animation-duration:24s;animation-delay:-11s'></i><i style='left:54.6%;width:5px;height:5px;animation-duration:15s;animation-delay:-6s'></i><i style='left:67.3%;width:7px;height:7px;animation-duration:19s;animation-delay:-18s'></i><i style='left:45.5%;width:6px;height:6px;animation-duration:27s;animation-delay:-21s'></i><i style='left:25.4%;width:9px;height:9px;animation-duration:22s;animation-delay:-26s'></i><i style='left:72.0%;width:6px;height:6px;animation-duration:30s;animation-delay:-4s'></i><i style='left:42.1%;width:10px;height:10px;animation-duration:16s;animation-delay:-15s'></i><i style='left:5.8%;width:9px;height:9px;animation-duration:26s;animation-delay:-17s'></i><i style='left:86.0%;width:7px;height:7px;animation-duration:25s;animation-delay:-18s'></i><i style='left:57.7%;width:8px;height:8px;animation-duration:27s;animation-delay:-28s'></i><i style='left:47.5%;width:9px;height:9px;animation-duration:15s;animation-delay:-21s'></i><i style='left:64.1%;width:12px;height:12px;animation-duration:27s;animation-delay:-9s'></i><i style='left:39.0%;width:9px;height:9px;animation-duration:14s;animation-delay:-14s'></i><i style='left:18.1%;width:5px;height:5px;animation-duration:15s;animation-delay:-23s'></i><i style='left:14.4%;width:6px;height:6px;animation-duration:20s;animation-delay:-26s'></i>
<svg class="floor" viewBox="0 0 1440 120" preserveAspectRatio="none"><path d="M0 90 C120 70 220 100 340 84 C470 66 560 96 700 88 C850 78 930 58 1060 76 C1180 92 1300 70 1440 82 V120 H0Z" fill="#031019"/>
<path d="M0 104 C160 92 300 112 480 100 C660 90 820 112 1000 102 C1180 92 1320 108 1440 100 V120 H0Z" fill="#020B12"/></svg></div>
<div class="gauge"><div class="ttl">DEPTH</div><div class='tk' style='top:0%'><span>0 m</span></div><div class='tk' style='top:25%'><span>50 m</span></div><div class='tk' style='top:50%'><span>100 m</span></div><div class='tk' style='top:75%'><span>150 m</span></div><div class='tk' style='top:100%'><span>200 m</span></div><div class="mk"></div></div>
<div class="sonar"><span></span><span></span><span></span></div>
""", unsafe_allow_html=True)


# ---------- loading ----------
BOILER = re.compile(r"no recommendation|makes no|has no|not make any|^\W*n/?a\b|specific nature of this incident|"
                    r"recommendations? (to|for) (the )?(mms|bsee|regional|agency)|agrees with the procedure|course of action for mms|"
                    r"regional office of safety|does not have|\bn/a\b|office of incident|^\W*none\W*$", re.I)


def clean_recs(recs):
    """Keep only actionable recommendation text: drop administrative boilerplate sentences."""
    out = []
    for r in recs:
        sents = re.split(r"(?<=[.;])\s+", r.lstrip("$-• "))
        keep = " ".join(s for s in sents if not BOILER.search(s)).strip()
        if len(keep) >= 30:
            out.append(keep)
    return out


@st.cache_resource
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    e = np.array(list(embedder().embed(texts, batch_size=32)), dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


@st.cache_resource
def load():
    inc = pd.read_pickle(ART / "incidents.pkl")
    emb = np.load(ART / "emb.npy")
    models = pickle.load(open(ART / "models.pkl", "rb"))
    ev = json.load(open(ART / "eval.json"))
    oof = pd.read_csv(ART / "oof.csv", dtype={"y": str})
    gal = pd.read_pickle(ART / "galaxy.pkl")
    inc = inc.merge(gal, on="id", how="left")
    inc["recs_raw"] = inc.recs
    inc["recs"] = inc.recs.map(clean_recs)
    return inc, emb, models, ev, oof


inc, EMB, M, EV, OOF = load()
THEMES = inc.theme.value_counts().index.tolist()
PALETTE = ["#2DD4BF", "#38BDF8", "#A78BFA", "#FBBF24", "#FF7A45", "#F472B6", "#5EEAD4", "#60A5FA", "#C084FC",
           "#34D399", "#94A3B8", "#FDA4AF"]
TCOL = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(THEMES)}
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA")


def galaxy_fig(highlight=None, star=None, height=560, legend=True):
    fig = go.Figure()
    for t in THEMES:
        d = inc[inc.theme == t]
        fig.add_trace(go.Scattergl(
            x=d.x, y=d.y, mode="markers", name=f"{t} ({len(d)})",
            marker=dict(size=6 if highlight is None else 4, color=TCOL[t], opacity=0.85 if highlight is None else 0.25,
                        line=dict(width=0)),
            customdata=np.stack([d.type, d.date.str[:4], d.what.str.lstrip("-• ").str[:140]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}…<extra>" + t + "</extra>"))
    if highlight is not None:
        h = inc.iloc[highlight]
        fig.add_trace(go.Scatter(x=h.x, y=h.y, mode="markers", name="Look-alikes",
                                 marker=dict(size=13, color="#F59E0B", line=dict(width=2, color="#FFFFFF")), hoverinfo="skip"))
    if star is not None:
        fig.add_trace(go.Scatter(x=[star[0]], y=[star[1]], mode="markers+text", name="New report", text=["  NEW REPORT"],
                                 textposition="middle right", textfont=dict(color="#FFFFFF", size=13),
                                 marker=dict(size=24, symbol="star", color="#FFFFFF", line=dict(width=2, color="#F59E0B"))))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=50), showlegend=legend,
                      legend=dict(font=dict(size=11), itemsizing="constant", bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), hoverlabel=dict(bgcolor="#0B2233"), **DARK)
    return fig


def short(t, n):
    """Trim to n chars, preferring a sentence end."""
    if len(t) <= n:
        return t
    cut = t[:n]
    i = cut.rfind(". ")
    return cut[:i + 1] if i > 60 else cut.rstrip() + "…"


def alert_html(q, res, tally, recs, watch):
    """Safety alert: deep-ocean briefing card on screen, high-contrast white sheet when printed."""
    items = "".join(f"<li><div class='mt'><b>{esc(r.type) or 'Incident'} · {esc(r.date[:4]) or 'undated'}</b><span class='pill'>{r.sim*100:.0f}% match</span></div>"
                    f"{esc(r.what.lstrip('-• ')[:240])}…</li>" for _, r in res.head(3).iterrows())
    total = max(tally.values()) if tally else 1
    causes = "".join(f"<div class='bar'><span>{esc(k)}</span><i><u style='width:{100*v/total:.0f}%'></u></i><b>{v}</b></div>" for k, v in tally.items())
    fixes = "".join(f"<li><span class='box'></span><span>{esc(short(r, 240))}</span></li>" for r in recs[:5])
    watch_li = "".join(f"<li><span class='box'></span><span>{esc(short(w, 220))}</span></li>" for w in watch)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Safety Alert</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
html{{background:#071E2F}}
body{{margin:0 auto;max-width:860px;padding:22px;font:15px/1.5 Inter,Arial,sans-serif;color:#D5EEEA;
  background:radial-gradient(900px 400px at 85% -10%,rgba(255,122,69,.10),transparent 60%),#071E2F}}
.head{{position:relative;overflow:hidden;border-radius:18px;padding:22px 26px 20px 26px;display:flex;gap:20px;align-items:center;
  background:linear-gradient(120deg,rgba(255,122,69,.20),rgba(14,58,85,.85) 55%);border:1px solid rgba(255,122,69,.45);
  box-shadow:0 0 40px rgba(255,122,69,.15)}}
.head:before{{content:"";position:absolute;left:0;right:0;top:0;height:6px;
  background:repeating-linear-gradient(-45deg,#FF7A45 0 12px,#071E2F 12px 24px)}}
.tri{{width:64px;height:58px;flex:none;filter:drop-shadow(0 0 12px rgba(255,122,69,.6))}}
.eb{{font:600 11px 'JetBrains Mono',monospace;letter-spacing:.22em;color:#FFB08F}}
h1{{margin:4px 0 0;font:800 40px/1 Inter,Arial,sans-serif;letter-spacing:-.02em;color:#fff}}
.sub{{margin-top:6px;font-size:15px;color:#FFD3BF}}
.meta{{display:flex;justify-content:space-between;font-size:12px;color:#7FA7B5;margin:12px 4px 6px}}
.meta b{{color:#5EEAD4}}
section{{position:relative;margin:12px 0;padding:16px 20px 12px 66px;border-radius:16px;
  background:linear-gradient(180deg,rgba(16,58,82,.75),rgba(10,40,62,.65));border:1px solid rgba(94,234,212,.14)}}
section h2{{margin:0 0 8px;font:700 13px Inter,Arial,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#5EEAD4}}
.n{{position:absolute;left:18px;top:14px;width:32px;height:32px;border-radius:50%;display:grid;place-items:center;
  font:700 14px 'JetBrains Mono',monospace;color:#04202A;background:#2DD4BF;box-shadow:0 0 16px rgba(45,212,191,.55)}}
.new{{border-color:rgba(255,122,69,.4);background:linear-gradient(180deg,rgba(255,122,69,.10),rgba(10,40,62,.65))}}
.new .n{{background:#FF7A45;box-shadow:0 0 16px rgba(255,122,69,.6)}} .new h2{{color:#FFB08F}}
.new p{{margin:0;font-size:16px;color:#fff}}
ul{{margin:0;padding:0;list-style:none}} li{{margin:9px 0}}
.inc li{{padding-left:12px;border-left:2px solid rgba(94,234,212,.4);color:#BFE3DD}}
.mt{{display:flex;justify-content:space-between;gap:10px;align-items:center}} .mt b{{color:#fff}}
.pill{{flex:none;font:600 11px 'JetBrains Mono',monospace;color:#04202A;background:#5EEAD4;border-radius:99px;padding:2px 9px}}
.chk li{{display:flex;gap:12px;align-items:flex-start;color:#E6F4F1}}
.box{{flex:none;width:18px;height:18px;margin-top:2px;border-radius:5px;border:2px solid #2DD4BF;box-shadow:0 0 8px rgba(45,212,191,.35)}}
.bar{{display:grid;grid-template-columns:170px 1fr 28px;gap:12px;align-items:center;margin:7px 0;font-size:14px}}
.bar i{{display:block;height:10px;border-radius:99px;background:rgba(94,234,212,.08);overflow:hidden}}
.bar u{{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#14776E,#5EEAD4);box-shadow:0 0 10px rgba(45,212,191,.5)}}
.bar b{{text-align:right;font-family:'JetBrains Mono',monospace;color:#fff}}
.note{{font-size:12px;color:#7FA7B5;margin-top:6px}}
.sign{{display:flex;gap:28px;font-size:12px;color:#7FA7B5;margin:18px 6px 8px}} .sign div{{flex:1;border-top:1px solid rgba(94,234,212,.35);padding-top:6px}}
.foot{{font-size:11.5px;color:#6E8F9C;margin:10px 4px}}
@media print{{
  html,body{{background:#fff !important;color:#111}} body{{padding:0}}
  .head{{background:#FFD60A;border:3px solid #111;box-shadow:none}} .head:before{{background:repeating-linear-gradient(-45deg,#111 0 12px,#FFD60A 12px 24px)}}
  h1,.sub,.eb{{color:#111}} .tri{{filter:none}} .tri path:first-child{{fill:#111}}
  section{{background:#fff;border:2px solid #111;break-inside:avoid}} section h2,.new h2{{color:#111}} .new{{background:#FFF8D6}}
  .n,.new .n{{background:#111;color:#FFD60A;box-shadow:none}} .new p,.mt b,.chk li,.inc li{{color:#111}}
  .inc li{{border-left:3px solid #FFD60A}} .pill{{background:#eee;color:#111}} .box{{border-color:#111;box-shadow:none}}
  .bar u{{background:#111;box-shadow:none}} .bar b{{color:#111}} .meta,.note,.foot,.sign{{color:#444}} .meta b{{color:#0E7C70}}
  .sign div{{border-top:2px solid #111}}
}}
</style></head><body>
<div class="head"><svg class="tri" viewBox="0 0 100 88"><path d="M50 4 L96 84 H4 Z" fill="#FF7A45"/><path d="M50 20 L82 77 H18 Z" fill="#071E2F" opacity=".92"/>
<rect x="45" y="37" width="10" height="23" rx="3" fill="#FF7A45"/><circle cx="50" cy="68" r="5.5" fill="#FF7A45"/></svg>
<div><div class="eb">PRE-JOB BRIEFING · INCIDENT LENS</div><h1>Safety Alert</h1><div class="sub">This has happened before. Read before the job.</div></div></div>
<div class="meta"><span>Generated by <b>Incident Lens</b> from {len(inc):,} past offshore incident reports (public BSEE)</span><span>Print for a paper copy</span></div>
<section class="new"><div class="n">1</div><h2>The new report</h2><p>{esc(q)}</p></section>
<section class="inc"><div class="n">2</div><h2>Similar past incidents</h2><ul>{items}</ul></section>
<section><div class="n">3</div><h2>What caused them</h2>{causes or 'No causes recorded'}<div class="note">Cause categories predicted by our model for the causes of these look-alikes.</div></section>
<section class="chk"><div class="n">4</div><h2>Watch for</h2><ul>{watch_li or '<li>No causes recorded</li>'}</ul></section>
<section class="chk"><div class="n">5</div><h2>Fixes investigators recommended</h2><ul>{fixes or '<li><span class="box"></span><span>No closely matching fix on record. Discuss controls at the pre-job meeting.</span></li>'}</ul></section>
<div class="sign"><div>Briefed by</div><div>Crew signatures</div><div>Date</div></div>
<p class="foot">Checklist items are what investigators found and recommended for past incidents. A decision aid for supervisors,
not a substitute for a job safety analysis.</p></body></html>"""


NAMES = M["names"]
CLF = M["lr_emb"]


def esc(t):
    return html.escape(t or "")


def predict(texts):
    P = CLF.predict_proba(embed(texts))
    return [{NAMES[c]: float(p) for c, p in zip(CLF.classes_, row)} for row in P]


# ---------- header ----------
best = EV["results"][EV["best"]]
base = EV["results"]["Baseline: TF-IDF + LR (official)"]
n_fix = int((inc.recs.str.len() > 0).sum())
HERO_BG = "data:image/jpeg;base64," + __import__("base64").b64encode(open(Path(__file__).parent / "assets_hero_galaxy.jpg", "rb").read()).decode()
st.markdown(f"<style>.hero:after{{background-image:url({HERO_BG});}}</style>", unsafe_allow_html=True)
st.markdown(f"""<div class="hero"><div class="eyebrow">INCIDENT LENS · OFFSHORE SAFETY INTELLIGENCE</div>
<h1>Every offshore incident,<br><em>so the next one doesn't happen.</em></h1>
<p>Search {len(inc):,} real BSEE incident reports by meaning, see what caused look-alikes and what investigators recommended,
and turn it into a printable safety alert in one click.</p>
<div class="kpis">
<div><div class="v count" style="--t:{len(inc)}"></div><div class="l">real reports</div></div>
<div><div class="v count" style="--t:{len(THEMES)}"></div><div class="l">themes discovered</div></div>
<div><div class="v">{best['macro_f1']:.2f}</div><div class="l">cause macro-F1 (baseline {base['macro_f1']:.2f})</div></div>
<div><div class="v" style="color:#FF7A45"><span class="count" style="--t:{round(100*n_fix/len(inc))}"></span>%</div><div class="l">reports with a real fix</div></div>
</div>
<div style="margin-top:18px;position:relative"><span class="chip">Track 9 · Local Business &amp; Industry</span>
<span class="chip">Real report text only · synthetic cells excluded</span>
<span class="chip">Runs fully local · no data leaves the laptop</span></div></div>""", unsafe_allow_html=True)

TAB_NAMES = [":material/bubble_chart: Incident galaxy", ":material/travel_explore: Similar incidents", ":material/psychology: Cause & why",
             ":material/fact_check: Report gaps & trends", ":material/insights: How well it works"]
_sel = st.query_params.get("tab", "0")
_order = [int(_sel)] + [i for i in range(5) if i != int(_sel)] if _sel in "01234" and len(_sel) == 1 else list(range(5))
_ts = st.tabs([TAB_NAMES[i] for i in _order])
tab0, tab1, tab2, tab3, tab4 = (_ts[_order.index(i)] for i in range(5))

# ---------- TAB 0: galaxy ----------
with tab0:
    st.markdown(f"<div class='lede'>1,212 incidents. <em>12 patterns that keep coming back.</em></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>01 · THE MAP</div>", unsafe_allow_html=True)
    st.markdown("#### Every incident, placed by meaning")
    st.caption(f"Each dot is one of {len(inc):,} real reports. Reports that describe similar events sit close together. "
               "Colors are 12 themes the model discovered on its own (k-means on the embeddings; we named them by their top words). Hover a dot to read it.")
    components.html(galaxy_html(inc, THEMES, TCOL, height=640, title_note=f"{len(inc):,} real reports · 12 themes found by the model · hover any star · drag to spin"), height=650)
    cols = st.columns(4)
    gx0, gx1, gy0, gy1 = inc.x.min(), inc.x.max(), inc.y.min(), inc.y.max()
    for i, t in enumerate(THEMES):
        d = inc[inc.theme == t]
        bgdots = "".join(f"<circle cx='{(x_-gx0)/(gx1-gx0)*200:.1f}' cy='{(y_-gy0)/(gy1-gy0)*64:.1f}' r='0.9' fill='#3E6A7A'/>"
                         for x_, y_ in zip(inc.x.values[::4], inc.y.values[::4]))
        dots = "".join(f"<circle cx='{(x_-gx0)/(gx1-gx0)*200:.1f}' cy='{(y_-gy0)/(gy1-gy0)*64:.1f}' r='1.8' fill='{TCOL[t]}'/>"
                       for x_, y_ in zip(d.x.values[:90], d.y.values[:90]))
        mini = (f"<svg class='mini' viewBox='-4 -4 208 72' preserveAspectRatio='none'><defs><filter id='gl{i}'><feGaussianBlur stdDeviation='1.6'/></filter></defs>"
                f"{bgdots}<g filter='url(#gl{i})'>{dots}</g>{dots}</svg>")
        cols[i % 4].markdown(f"<div class='card' style='padding:12px 14px'><span style='color:{TCOL[t]};font-weight:700'>● {t}</span>{mini}"
                             f"<div class='big' style='font-size:1.6rem;margin-top:4px'>{len(d)}</div>"
                             f"<div class='muted'>{esc(d.keywords.iloc[0])}</div></div>", unsafe_allow_html=True)

# ---------- TAB 1: search ----------
EXAMPLES = {
    "Dropped object during crane lift": "During a crane lift of a bundle of pipe from the supply vessel, the sling parted and the load fell to the deck, striking a roustabout on the leg.",
    "Hand caught while making up pipe": "While making up drill pipe on the rig floor, the floorhand's hand was caught between the tongs and the pipe, resulting in a fractured finger.",
    "Gas release from leaking flange": "A gas release occurred when a flange on the separator outlet leaked during startup. Gas detectors alarmed and the platform was shut in.",
    "Slip on wet stairs": "A worker slipped while descending wet stairs between decks during heavy rain and injured his back.",
}
with tab1:
    st.markdown(f"<div class='lede'>A new report comes in. <em>What happened before, and what fixed it?</em></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c2:
        st.markdown("**Try an example**")
        pick = st.radio("example", list(EXAMPLES), label_visibility="collapsed")
    with c1:
        q = st.text_area("Paste a new incident report", EXAMPLES[pick], height=130)
    k = st.slider("How many look-alikes", 3, 10, 5)
    if q.strip():
        qe = embed([q])[0]
        sims = EMB @ qe
        top = np.argsort(-sims)[:k]
        res = inc.iloc[top].assign(sim=sims[top])

        star = (float(res.x.mean()), float(res.y.mean()))
        st.markdown("<div class='sec'>WHERE THIS REPORT LANDS</div>", unsafe_allow_html=True)
        components.html(galaxy_html(inc, THEMES, TCOL, height=430, target=star, neighbors=top, sims=sims[top],
                                    title_note=f"New report vs {len(inc):,} past incidents · lines show the {k} closest"), height=440)
        left, right = st.columns([3, 2])
        with right:
            st.markdown("#### What past incidents teach")
            all_causes = [c for cs in res.causes for c in cs]
            if all_causes:
                preds = predict(all_causes)
                tally = pd.Series([max(p, key=p.get) for p in preds]).value_counts()
                fig = go.Figure(go.Bar(x=tally.values, y=tally.index, orientation="h",
                                       marker_color=[COLORS.get(i, "#888") for i in tally.index]))
                fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=50), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA", xaxis_title="cause statements in look-alikes")
                st.plotly_chart(fig, use_container_width=True, config=PCFG)
                st.caption("Cause categories predicted by our model for the causes of these look-alike incidents.")
            st.markdown("**Fixes that were recommended before**")
            wide = inc.iloc[np.argsort(-sims)[:40]]
            cand = list(dict.fromkeys(r for rs in wide.recs for r in rs))[:80]
            if cand:
                rs_ = embed(cand) @ qe
                recs = [cand[i] for i in np.argsort(-rs_)[:5] if rs_[i] >= 0.60]
            else:
                recs = []
            st.caption("From the 40 closest incidents: actionable fixes that closely match this report (boilerplate removed; "
                       f"only {n_fix} of {len(inc):,} reports contain an actionable fix at all).")
            for r in recs:
                st.markdown(f"<div class='lesson'>{esc(r[:260])}</div>", unsafe_allow_html=True)
            if not recs:
                st.caption("No recommendations recorded for these incidents.")
            tally_s = tally.to_dict() if all_causes else {}
            watch = [c.strip() for cs in res.causes for c in cs if c.strip()[:1].isupper() and c.strip().endswith((".", ")"))]
            html_doc = alert_html(q, res, tally_s, recs, list(dict.fromkeys(watch))[:6])
            if st.button("⚠️  Generate safety alert", type="primary", use_container_width=True):
                st.session_state["alert"] = html_doc
                st.session_state["alert_q"] = q
            if st.session_state.get("alert") and st.session_state.get("alert_q") == q:
                st.download_button("⬇  Download alert (HTML, printable)", st.session_state["alert"], "safety_alert.html",
                                   "text/html", use_container_width=True)
        with left:
            st.markdown("#### Most similar past incidents")
            for _, r in res.iterrows():
                causes = "".join(f"<div class='cause'>{esc(c[:240])}</div>" for c in r.causes[:3]) or "<span class='muted'>No cause recorded</span>"
                st.markdown(f"""<div class="card"><div class="ring" style="--p:{r.sim*100:.0f}"><span>{r.sim*100:.0f}%<small>MATCH</small></span></div>
<h4>{esc(r.type) or 'Incident'} · {esc(r.date[:4]) or 'undated'}</h4>
<div class="muted">{esc(r.event) or ''} {('· area ' + esc(r.area)) if r.area else ''}</div>
<p style="margin:8px 0">{esc(r.what.lstrip('-• ')[:420])}{'…' if len(r.what) > 420 else ''}</p>
<div class="muted" style="margin-bottom:2px">Causes found by the investigator</div>{causes}</div>""", unsafe_allow_html=True)
        st.caption("Similarity = cosine similarity of meaning (bge-small embeddings), not keyword overlap. "
                   "Look-alikes are a starting point for an investigator, not a conclusion.")
    if st.session_state.get("alert") and st.session_state.get("alert_q") == q:
        st.markdown("<div class='sec'>SAFETY ALERT PREVIEW</div>", unsafe_allow_html=True)
        components.html(st.session_state["alert"], height=1100, scrolling=True)

# ---------- TAB 2: cause ----------
CAUSE_EX = [
    "The crane operator did not follow the lifting procedure and failed to verify the load weight before the lift.",
    "The hydraulic hose on the BOP control line ruptured due to internal corrosion.",
    "Radio instructions between the deck crew and the crane operator were not understood.",
    "No job safety analysis was performed for the non-routine task.",
    "The person in charge did not ensure the work permit was reviewed before starting the job.",
    "Heavy seas and high winds caused the personnel basket to swing into the handrail.",
]
with tab2:
    st.markdown(f"<div class='lede'>Which cause category would an investigator pick, <em>and which words gave it away?</em></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c2:
        st.markdown("**Try a cause statement**")
        ex = st.selectbox("ex", CAUSE_EX, label_visibility="collapsed")
    with c1:
        txt = st.text_area("Cause statement from an investigation", ex, height=110)
    if txt.strip():
        p = predict([txt])[0]
        order = sorted(p, key=p.get, reverse=True)
        top_c, top_p = order[0], p[order[0]]
        a, b = st.columns([2, 3])
        with a:
            st.markdown(f"<div class='kpi'><div class='big' style='color:{COLORS[top_c]}'>{top_c}</div>"
                        f"<div class='lbl'>most likely cause category · {top_p*100:.0f}% model confidence</div></div>",
                        unsafe_allow_html=True)
            if top_p < 0.45:
                st.warning("Low confidence: two or more categories are close. An investigator should decide.")
            rows_ = "".join(f"<div class='bar{' top' if j == 0 else ''}'><span>{o}</span><div class='tr'><div class='fl' style='--w:{p[o]*100:.1f}%;--c:{COLORS[o]};animation-delay:{j*90}ms'></div></div><b>{p[o]*100:.0f}%</b></div>"
                            for j, o in enumerate(order))
            st.markdown(f"<div class='bars'>{rows_}</div>", unsafe_allow_html=True)
        with b:
            st.markdown("#### Why: which words pushed the answer")
            words = re.findall(r"\S+", txt)[:60]
            variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
            cls_idx = list(CLF.classes_).index([c for c, n in NAMES.items() if n == top_c][0])
            P = CLF.predict_proba(embed(variants))[:, cls_idx]
            imp = np.clip(top_p - P, 0, None)
            imp = imp / imp.max() if imp.max() > 0 else imp
            col = COLORS[top_c].lstrip("#")
            rgb = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4))
            spans = " ".join(f"<span class='w' style='background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.06 + 0.7 * v:.2f});"
                             f"box-shadow:0 0 {v*22:.0f}px rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.65 * v:.2f});"
                             f"color:{'#04202A' if v > 0.6 else '#E6F4F1'};font-weight:{700 if v > 0.45 else 400}'>{esc(w)}</span>"
                             for w, v in zip(words, imp))
            st.markdown(f"<div class='card' style='font-size:1.08rem; line-height:2'>{spans}</div>", unsafe_allow_html=True)
            st.caption("Each word is removed in turn; the brighter the highlight, the more the confidence in "
                       f"“{top_c}” drops without it (leave-one-out explanation).")
            st.markdown("#### Precedents: closest labeled past causes")
            E = M["train_E"]; qe = embed([txt])[0]
            nn = np.argsort(-(E @ qe))[:3]
            for i in nn:
                lab = NAMES[M["train_y"][i]]
                st.markdown(f"<div class='cause' style='border-color:{COLORS[lab]}'><b style='color:{COLORS[lab]}'>{lab}</b> · "
                            f"{esc(M['train_text'][i][:220])}</div>", unsafe_allow_html=True)

# ---------- TAB 3: gaps ----------
n_nofix_total = int((inc.recs.str.len() == 0).sum())
with tab3:
    st.markdown(f"<div class='statbig'><div class='n'>{n_nofix_total}</div><div><div class='t'>of {len(inc):,} offshore incident reports "
                f"<em>never wrote down a fix anyone could act on.</em></div><div class='s'>Investigators found causes in almost every report, "
                f"but the recommendation field is mostly boilerplate such as \u201cno recommendations\u201d. The lessons exist; they are not written down.</div></div></div>",
                unsafe_allow_html=True)
    g = inc.copy()
    g["No cause recorded"] = g.causes.str.len() == 0
    g["No actionable fix"] = g.recs.str.len() == 0
    g["No outcome stated"] = g.outcome.str.len() == 0
    g["Short narrative (<300 chars)"] = g.what.str.len() < 300
    g["No date"] = g.date.str.len() == 0
    checks = ["No cause recorded", "No actionable fix", "No outcome stated", "Short narrative (<300 chars)", "No date"]
    g["gaps"] = g[checks].sum(axis=1)
    cols = st.columns(len(checks))
    for c, name in zip(cols, checks):
        c.markdown(f"<div class='kpi'><div class='big'>{int(g[name].sum())}</div><div class='lbl'>{name}</div></div>",
                   unsafe_allow_html=True)
    st.write("")
    yr = g.dropna(subset=["year"]).groupby("year")[["No actionable fix"]].mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yr.index, y=yr["No actionable fix"], name="No actionable fix", mode="lines+markers",
                             line=dict(color=AC, width=3, shape="spline"), marker=dict(size=7, color="#5EEAD4"),
                             fill="tozeroy", fillcolor="rgba(45,212,191,.12)", hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
    fig.update_layout(height=280, title="Share of reports with no actionable fix, by year (%)", showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#D5EEEA", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True, config=PCFG)
    tr = inc.dropna(subset=["year"]).query("year >= 2000").groupby(["year", "theme"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for t in THEMES:
        if t in tr:
            fig.add_trace(go.Bar(x=tr.index, y=tr[t], name=t, marker_color=TCOL[t]))
    fig.update_layout(barmode="stack", height=340, title="Incidents per year by theme (in this dataset)",
                      margin=dict(l=10, r=10, t=40, b=50), legend=dict(font=dict(size=10)), **DARK)
    st.plotly_chart(fig, use_container_width=True, config=PCFG)
    st.caption("Counts reflect which reports BSEE published and this dataset includes, not the full offshore incident rate.")
    sel = st.multiselect("Show reports missing…", checks, default=["No cause recorded"])
    view = g[g[sel].any(axis=1)] if sel else g[g.gaps > 0]
    st.caption(f"{len(view)} reports flagged. Flags come only from fields in the real report; a blank can also mean the source PDF did not carry that field.")
    st.dataframe(view.sort_values("gaps", ascending=False)[["date", "type", "event", "gaps"] + checks].head(300),
                 use_container_width=True, hide_index=True, height=330)

# ---------- TAB 4: honesty ----------
with tab4:
    st.markdown(f"<div class='lede'>0.48 vs 0.35, <em>measured exactly the way the official baseline was.</em></div>", unsafe_allow_html=True)
    st.markdown("#### Task: recover the investigator's cause category from the cause text")
    st.markdown(f"<span class='chip'>n = {EV['n']} cause statements</span><span class='chip'>{EV['n_incidents']} incidents</span>"
                "<span class='chip'>2019+ era only (one consistent label set)</span>"
                "<span class='chip'>5 folds grouped by incident (no leakage)</span>"
                "<span class='chip'>95% CI: 1,000 bootstrap resamples over incidents</span>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    maj = EV["results"]["Majority class (always guess most common)"]
    k1.markdown(f"<div class='kpi'><div class='big'>{maj['macro_f1']:.2f}</div><div class='lbl'>Always guess the most common class</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><div class='big'>{base['macro_f1']:.2f}</div><div class='lbl'>Official baseline (TF-IDF word counts)</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><div class='big' style='color:#2DD4BF'>{best['macro_f1']:.2f}</div><div class='lbl'>Incident Lens (meaning embeddings)</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        names = [n for n in EV["results"] if EV["results"][n]["ci"]]
        fig = go.Figure()
        for n in names:
            r = EV["results"][n]
            fig.add_trace(go.Scatter(x=[r["macro_f1"]], y=[n], mode="markers", marker=dict(size=14, color=AC if n == EV["best"] else GREY),
                                     error_x=dict(type="data", symmetric=False, array=[r["ci"][1] - r["macro_f1"]], arrayminus=[r["macro_f1"] - r["ci"][0]]),
                                     showlegend=False))
        fig.update_layout(title="Macro-F1 with 95% interval (no overlap = real difference)", height=300,
                          margin=dict(l=10, r=10, t=40, b=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#D5EEEA", xaxis_range=[0.2, 0.6])
        st.plotly_chart(fig, use_container_width=True, config=PCFG)
    with b:
        labs = EV["labels"]
        fig = go.Figure([go.Bar(name="Baseline", x=labs, y=[base["per_class"][l] for l in labs], marker_color=GREY),
                         go.Bar(name="Incident Lens", x=labs, y=[best["per_class"][l] for l in labs], marker_color=AC)])
        fig.update_layout(title="F1 per category: biggest gains on the rare ones", barmode="group", height=330,
                          margin=dict(l=10, r=10, t=70, b=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#D5EEEA", legend=dict(orientation="h", y=1.12, x=0), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    c, d = st.columns(2)
    with c:
        cm = np.array(EV["confusion"])
        n_ = len(labs); diag = np.eye(n_, dtype=bool)
        zc = np.where(diag, 0.56 + 0.44 * cm / cm.max(), np.where(cm >= 10, 0.32 + 0.12 * np.minimum((cm - 10) / 8, 1), 0.05 + 0.1 * cm / 10))
        fig = go.Figure(go.Heatmap(z=zc, x=labs, y=labs, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
                                   colorscale=[[0, "#0B2B40"], [0.15, "#13384F"], [0.3, "#B4532A"], [0.44, "#FF7A45"], [0.5, "#0B2B40"],
                                               [0.56, "#0F4C5C"], [1, "#5EEAD4"]],
                                   customdata=cm, hovertemplate="Investigator: %{y}<br>Model: %{x}<br>%{customdata} statements<extra></extra>"))
        for i_ in range(n_):
            for j_ in range(n_):
                v_ = int(cm[i_, j_])
                fig.add_annotation(x=labs[j_], y=labs[i_], text=f"<b>{v_}</b>" if (i_ == j_ or v_ >= 10) else str(v_), showarrow=False,
                                   font=dict(size=13, color="#04202A" if zc[i_, j_] > 0.8 else ("#FFFFFF" if v_ >= 10 or i_ == j_ else "#6E8F9C")))
        fig.update_layout(title="Where it gets confused (rows = investigator, cols = model)", height=430,
                          margin=dict(l=10, r=10, t=40, b=50), paper_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA",
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config=PCFG)
        st.markdown("<span class='muted'>Teal diagonal = correct. <b style='color:#FF7A45'>Orange</b> = a mistake made 10+ times.</span>", unsafe_allow_html=True)
    with d:
        st.markdown("#### Check against our own hand labels")
        GJ = ART / "gold.json"
        gj = json.load(open(GJ)) if GJ.exists() else {"n": 0}
        if gj.get("n"):
            st.markdown(f"<div class='kpi'><div class='big'>{gj['agree']} / {gj['n']}</div>"
                        f"<div class='lbl'>hand-labeled causes where the model matches our team ({gj['agree_rate']*100:.0f}%, "
                        f"95% CI {gj['ci95'][0]*100:.0f}–{gj['ci95'][1]*100:.0f}%)</div></div>", unsafe_allow_html=True)
            st.caption(f"{gj['n']} labeled rows from {gj['n_incidents']} incidents ({gj['n_question']} marked “?” excluded). "
                       "Every prediction comes from a model that never saw that incident. These labels were not used to tune the model. "
                       f"Our team matched the original investigators on {gj['team_vs_investigator']['agree']} of {gj['team_vs_investigator']['n']} rows, "
                       "so the categories are ambiguous for people too.")
            dis = HERE_LAB / "gold_disagreements.csv"
            if dis.exists():
                dd = pd.read_csv(dis, encoding="utf-8-sig").head(4)
                for _, r in dd.iterrows():
                    st.markdown(f"<div class='cause'><b>We said:</b> {esc(r.team)} · <b>Model:</b> {esc(r.model_pred)}<br>"
                                f"<span class='muted'>{esc(str(r.text)[:200])}</span></div>", unsafe_allow_html=True)
        else:
            st.info("Hand labels not imported yet (run gold_check.py).")
        st.markdown("#### Limits we know about")
        st.markdown("""
- Scores measure agreement with the **investigator's** category, not the "true" cause.
- Only 374 labeled rows; the rare categories have ~26–28 examples each, so their scores swing.
- Trained on the 2019+ label set only; older reports used different vocabularies.
- Model hyper-parameters were set once, not tuned on the test folds.
- Some real names may remain in source text; we scrub obvious ones and never show name fields.
- A decision aid for investigators, not a replacement for an investigation.""")
