"""Incident Lens — learn from offshore incident reports without reading every one.

Run: uv run streamlit run app.py
"""
from pathlib import Path
import json, pickle, re, html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from galaxy_component import galaxy_html

ART = Path(__file__).parent / "artifacts"
GOLD = Path(__file__).parent / "labeling" / "gold_to_label.csv"
CODE2EL = {"HU": "3", "EQ": "15", "MS": "8", "SU": "17", "CO": "9", "WE": "6"}
COLORS = {"Human performance": "#2DD4BF", "Equipment failure": "#38BDF8", "Management systems": "#A78BFA",
          "Supervision": "#F472B6", "Work environment": "#FBBF24", "Communication": "#FF7A45"}
AC, HOT, GREY = "#2DD4BF", "#FF7A45", "#3E5A6B"

st.set_page_config(page_title="Incident Lens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
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
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=legend,
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
    """Printable industrial hazard-poster style safety alert."""
    items = "".join(f"<li><b>{esc(r.type) or 'Incident'} · {esc(r.date[:4]) or 'undated'} · {r.sim*100:.0f}% match</b><br>{esc(r.what.lstrip('-• ')[:240])}…</li>"
                    for _, r in res.head(3).iterrows())
    total = sum(tally.values()) or 1
    causes = "".join(f"<div class='bar'><span>{esc(k)}</span><i style='width:{100*v/total:.0f}%'></i><b>{v}</b></div>" for k, v in tally.items())
    fixes = "".join(f"<li><span class='box'></span>{esc(short(r, 240))}</li>" for r in recs[:5])
    watch_li = "".join(f"<li><span class='box'></span>{esc(short(w, 220))}</li>" for w in watch)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Safety Alert</title><style>
html{{background:#fff}} *{{box-sizing:border-box}}
body{{font-family:'Arial Black',Arial,Helvetica,sans-serif;max-width:820px;margin:0 auto;padding:22px;color:#111;background:#fff}}
.stripe{{height:22px;background:repeating-linear-gradient(-45deg,#FFD60A 0 22px,#111 22px 44px);border-radius:4px}}
.head{{background:#FFD60A;border:6px solid #111;border-top:0;border-bottom:0;padding:18px 24px;display:flex;gap:20px;align-items:center}}
.tri{{width:88px;height:78px;flex:none}}
.head h1{{margin:0;font-size:46px;line-height:.95;letter-spacing:-1px}}
.head p{{margin:6px 0 0;font:700 17px Arial,sans-serif;text-transform:uppercase;letter-spacing:.06em}}
.meta{{display:flex;justify-content:space-between;font:12px Arial,sans-serif;color:#444;margin:10px 2px 4px}}
.meta b{{color:#0E7C70}}
section{{border:3px solid #111;border-radius:10px;padding:14px 18px 10px 62px;margin:14px 0;position:relative;font:15px/1.45 Arial,sans-serif}}
section h2{{font:900 16px 'Arial Black',Arial,sans-serif;text-transform:uppercase;margin:0 0 8px;letter-spacing:.03em}}
.n{{position:absolute;left:14px;top:12px;width:34px;height:34px;border-radius:50%;background:#111;color:#FFD60A;
  font:900 18px 'Arial Black',Arial;display:flex;align-items:center;justify-content:center}}
.new{{background:#FFF8D6}} ul{{margin:0;padding:0;list-style:none}} li{{margin:7px 0;display:flex;gap:10px}}
section:not(.chk) li{{display:block;padding-left:12px;border-left:4px solid #FFD60A}}
.box{{flex:none;width:18px;height:18px;border:2.5px solid #111;border-radius:3px;margin-top:1px}}
.bar{{display:grid;grid-template-columns:170px 1fr 30px;gap:10px;align-items:center;margin:6px 0}}
.bar i{{display:block;height:14px;background:#111;border-radius:2px}} .bar b{{text-align:right}}
.sign{{display:flex;gap:30px;font:13px Arial,sans-serif;margin:16px 4px 12px}} .sign div{{flex:1;border-top:2px solid #111;padding-top:4px}}
.foot{{font:11.5px Arial,sans-serif;color:#555;margin:10px 2px}}
@media print{{body{{padding:0}} section{{break-inside:avoid}}}}
</style></head><body>
<div class="stripe"></div>
<div class="head"><svg class="tri" viewBox="0 0 100 88"><path d="M50 4 L96 84 H4 Z" fill="#111"/><path d="M50 18 L84 78 H16 Z" fill="#FFD60A"/>
<rect x="45" y="36" width="10" height="24" rx="3" fill="#111"/><circle cx="50" cy="68" r="5.5" fill="#111"/></svg>
<div><h1>SAFETY ALERT</h1><p>This has happened before. Read before the job.</p></div></div>
<div class="stripe"></div>
<div class="meta"><span>Generated by <b>Incident Lens</b> from {len(inc):,} past offshore incident reports (public BSEE)</span><span>Pre-job briefing</span></div>
<section class="new"><div class="n">1</div><h2>The new report</h2>{esc(q)}</section>
<section><div class="n">2</div><h2>Similar past incidents</h2><ul>{items}</ul></section>
<section><div class="n">3</div><h2>What caused them</h2>{causes or 'No causes recorded'}<div class="foot" style="margin:6px 0 0">Cause categories predicted by our model for the causes of these look-alikes.</div></section>
<section class="chk"><div class="n">4</div><h2>Watch for</h2><ul>{watch_li or '<li>No causes recorded</li>'}</ul></section>
<section class="chk"><div class="n">5</div><h2>Fixes investigators recommended</h2><ul>{fixes or '<li><span class="box"></span>No closely matching fix on record. Discuss controls at the pre-job meeting.</li>'}</ul></section>
<div class="sign"><div>Briefed by</div><div>Crew signatures</div><div>Date</div></div>
<div class="stripe" style="height:12px"></div>
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

TAB_NAMES = ["🌌  Incident galaxy", "🔍  Similar incidents", "🧠  Cause & why", "⚠️  Report gaps & trends", "📊  How well it works"]
_sel = st.query_params.get("tab", "0")
_order = [int(_sel)] + [i for i in range(5) if i != int(_sel)] if _sel in "01234" and len(_sel) == 1 else list(range(5))
_ts = st.tabs([TAB_NAMES[i] for i in _order])
tab0, tab1, tab2, tab3, tab4 = (_ts[_order.index(i)] for i in range(5))

# ---------- TAB 0: galaxy ----------
with tab0:
    st.markdown("<div class='sec'>01 · THE MAP</div>", unsafe_allow_html=True)
    st.markdown("#### Every incident, placed by meaning")
    st.caption(f"Each dot is one of {len(inc):,} real reports. Reports that describe similar events sit close together. "
               "Colors are 12 themes the model discovered on its own (k-means on the embeddings; we named them by their top words). Hover a dot to read it.")
    components.html(galaxy_html(inc, THEMES, TCOL, height=640, title_note=f"{len(inc):,} real reports · 12 themes found by the model · hover any star"), height=650)
    cols = st.columns(4)
    for i, t in enumerate(THEMES[:8]):
        d = inc[inc.theme == t]
        cols[i % 4].markdown(f"<div class='card' style='padding:12px 14px'><span style='color:{TCOL[t]};font-weight:700'>● {t}</span>"
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
                fig.update_layout(height=230, margin=dict(l=0, r=10, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA", xaxis_title="cause statements in look-alikes")
                st.plotly_chart(fig, use_container_width=True)
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
                st.markdown(f"""<div class="card"><span class="sim">{r.sim*100:.0f}% match</span>
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
            fig = go.Figure(go.Bar(x=[p[o] for o in order][::-1], y=order[::-1], orientation="h",
                                   marker_color=[COLORS[o] for o in order][::-1]))
            fig.update_layout(height=250, margin=dict(l=0, r=10, t=10, b=0), xaxis_tickformat=".0%",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA")
            st.plotly_chart(fig, use_container_width=True)
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
            spans = " ".join(f"<span class='w' style='background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.08 + 0.75 * v:.2f})'>{esc(w)}</span>"
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
with tab3:
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
    yr = g.dropna(subset=["year"]).groupby("year")[["No cause recorded", "No actionable fix"]].mean() * 100
    fig = go.Figure()
    for cname, colr in [("No cause recorded", HOT), ("No actionable fix", AC)]:
        fig.add_trace(go.Scatter(x=yr.index, y=yr[cname], name=cname, mode="lines+markers", line_color=colr))
    fig.update_layout(height=280, title="Share of reports missing key details, by year (%)",
                      margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#D5EEEA", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    tr = inc.dropna(subset=["year"]).query("year >= 2000").groupby(["year", "theme"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for t in THEMES:
        if t in tr:
            fig.add_trace(go.Bar(x=tr.index, y=tr[t], name=t, marker_color=TCOL[t]))
    fig.update_layout(barmode="stack", height=340, title="Incidents per year by theme (in this dataset)",
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=10)), **DARK)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Counts reflect which reports BSEE published and this dataset includes, not the full offshore incident rate.")
    sel = st.multiselect("Show reports missing…", checks, default=["No cause recorded"])
    view = g[g[sel].any(axis=1)] if sel else g[g.gaps > 0]
    st.caption(f"{len(view)} reports flagged. Flags come only from fields in the real report; a blank can also mean the source PDF did not carry that field.")
    st.dataframe(view.sort_values("gaps", ascending=False)[["date", "type", "event", "gaps"] + checks].head(300),
                 use_container_width=True, hide_index=True, height=330)

# ---------- TAB 4: honesty ----------
with tab4:
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
                          margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#D5EEEA", xaxis_range=[0.2, 0.6])
        st.plotly_chart(fig, use_container_width=True)
    with b:
        labs = EV["labels"]
        fig = go.Figure([go.Bar(name="Baseline", x=labs, y=[base["per_class"][l] for l in labs], marker_color=GREY),
                         go.Bar(name="Incident Lens", x=labs, y=[best["per_class"][l] for l in labs], marker_color=AC)])
        fig.update_layout(title="F1 per category: biggest gains on the rare ones", barmode="group", height=330,
                          margin=dict(l=0, r=0, t=70, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#D5EEEA", legend=dict(orientation="h", y=1.12, x=0), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    with c:
        cm = np.array(EV["confusion"])
        fig = go.Figure(go.Heatmap(z=cm, x=labs, y=labs, colorscale=[[0, "#0B2233"], [0.5, "#14776E"], [1, "#5EEAD4"]], text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Where it gets confused (rows = investigator, cols = model)", height=380,
                          margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#D5EEEA",
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with d:
        st.markdown("#### Check against our own hand labels")
        if GOLD.exists():
            gd = pd.read_csv(GOLD, encoding="utf-8-sig")
            gd = gd[gd.my_label.astype(str).str.upper().str.strip().isin(CODE2EL)]
            if len(gd):
                gd["el"] = gd.my_label.str.upper().str.strip().map(CODE2EL)
                OOF["rid"] = OOF["Incident Number"] + "-" + OOF["Cause number"].astype(str)
                oofmap = dict(zip(OOF.rid, OOF[EV["best"]].astype(str)))
                inv = {n: c for c, n in NAMES.items()}
                mp = []
                for _, r in gd.iterrows():
                    if r.report_id in oofmap:   # seen in training -> use out-of-fold prediction
                        mp.append(oofmap[r.report_id])
                    else:
                        pr = predict([str(r.text)])[0]
                        mp.append(inv[max(pr, key=pr.get)])
                gd["model"] = mp
                agree = (gd.el == gd.model).mean()
                st.markdown(f"<div class='kpi'><div class='big'>{(gd.el == gd.model).sum()} / {len(gd)}</div>"
                            f"<div class='lbl'>rows where the model matches our team's label ({agree*100:.0f}%)</div></div>",
                            unsafe_allow_html=True)
                miss = gd[gd.el != gd.model].head(4)
                for _, r in miss.iterrows():
                    st.markdown(f"<div class='cause'><b>We said:</b> {NAMES[r.el]} · <b>Model:</b> {NAMES[r.model]}<br>"
                                f"<span class='muted'>{esc(str(r.text)[:200])}</span></div>", unsafe_allow_html=True)
            else:
                st.info("Hand labels not filled in yet (labeling/gold_to_label.csv).")
        st.markdown("#### Limits we know about")
        st.markdown("""
- Scores measure agreement with the **investigator's** category, not the "true" cause.
- Only 374 labeled rows; the rare categories have ~26–28 examples each, so their scores swing.
- Trained on the 2019+ label set only; older reports used different vocabularies.
- Model hyper-parameters were set once, not tuned on the test folds.
- Some real names may remain in source text; we scrub obvious ones and never show name fields.
- A decision aid for investigators, not a replacement for an investigation.""")
