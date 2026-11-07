"""Incident Lens — learn from offshore incident reports without reading every one.

Run: uv run streamlit run app.py
"""
from pathlib import Path
import json, pickle, re, html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ART = Path(__file__).parent / "artifacts"
GOLD = Path(__file__).parent / "labeling" / "gold_to_label.csv"
CODE2EL = {"HU": "3", "EQ": "15", "MS": "8", "SU": "17", "CO": "9", "WE": "6"}
COLORS = {"Human performance": "#F59E0B", "Equipment failure": "#38BDF8", "Management systems": "#A78BFA",
          "Supervision": "#F472B6", "Work environment": "#34D399", "Communication": "#FB7185"}

st.set_page_config(page_title="Incident Lens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {padding: 22px 28px; border-radius: 18px; margin-bottom: 18px;
  background: radial-gradient(1200px 300px at 10% -20%, rgba(245,158,11,.25), transparent),
              linear-gradient(135deg, #0F1A2E 0%, #13223D 100%);
  border: 1px solid rgba(255,255,255,.07);}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; color: #9CA3AF; font-size: 1.02rem;}
.chip {display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; margin: 0 6px 6px 0;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.09); color:#D1D5DB;}
.card {padding: 16px 18px; border-radius: 14px; background: #131C2E; border: 1px solid rgba(255,255,255,.07);
  margin-bottom: 12px;}
.card h4 {margin: 0 0 6px; font-size: 1.02rem;}
.sim {float:right; font-weight:700; color:#F59E0B;}
.muted {color:#9CA3AF; font-size:.88rem;}
.lesson {border-left: 3px solid #F59E0B; padding: 4px 10px; margin: 6px 0; background: rgba(245,158,11,.06); border-radius: 0 8px 8px 0;}
.cause {border-left: 3px solid #38BDF8; padding: 4px 10px; margin: 6px 0; background: rgba(56,189,248,.06); border-radius: 0 8px 8px 0;}
.big {font-size: 2.4rem; font-weight: 800; line-height: 1;}
.kpi {padding: 16px; border-radius: 14px; background:#131C2E; border:1px solid rgba(255,255,255,.07); text-align:center;}
.kpi .lbl {color:#9CA3AF; font-size:.82rem; margin-top:6px;}
.w {padding: 1px 2px; border-radius: 4px;}
@keyframes rise {from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;}}
.card, .kpi, .lesson, .cause {animation: rise .45s ease both;}
.card:hover {border-color: rgba(245,158,11,.45); transition: border-color .2s;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 10px 14px;}
.stTabs [aria-selected="true"] {color: #F59E0B !important;}
</style>""", unsafe_allow_html=True)


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
PALETTE = ["#F59E0B", "#38BDF8", "#A78BFA", "#34D399", "#FB7185", "#F472B6", "#FACC15", "#2DD4BF", "#818CF8",
           "#FB923C", "#94A3B8", "#C084FC"]
TCOL = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(THEMES)}
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB")


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
                      xaxis=dict(visible=False), yaxis=dict(visible=False), hoverlabel=dict(bgcolor="#131C2E"), **DARK)
    return fig


def alert_html(q, res, tally, recs, watch):
    items = "".join(f"<li><b>{esc(r.type) or 'Incident'} ({esc(r.date[:4]) or 'undated'})</b>: {esc(r.what.lstrip('-• ')[:260])}…</li>" for _, r in res.head(3).iterrows())
    causes = "".join(f"<li>{esc(k)}: {v} cause statement{'s' if v > 1 else ''}</li>" for k, v in tally.items())
    fixes = "".join(f"<li>☐ {esc(r[:220])}</li>" for r in recs[:5])
    watch_li = "".join(f"<li>☐ {esc(w[:200])}</li>" for w in watch)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Safety Alert</title><style>
html{{background:#fff}} body{{font-family:Calibri,Arial,sans-serif;max-width:780px;margin:0 auto;padding:24px;color:#111827;line-height:1.45;background:#fff}}
.top{{background:#0B1220;color:#fff;padding:18px 22px;border-radius:12px}} .top h1{{margin:0;font-size:26px}}
.top p{{margin:6px 0 0;color:#F59E0B}} h2{{font-size:17px;margin:22px 0 6px;color:#B45309}}
.box{{background:#F3F4F6;padding:10px 14px;border-radius:8px}} li{{margin:5px 0}} ul{{padding-left:20px}}
.foot{{color:#6B7280;font-size:12px;margin-top:24px}}</style></head><body>
<div class="top"><h1>⚠ Safety Alert: this has happened before</h1><p>Generated by Incident Lens from {len(inc):,} past offshore incident reports</p></div>
<h2>The new report</h2><div class="box">{esc(q)}</div>
<h2>Similar past incidents</h2><ul>{items}</ul>
<h2>What caused them (model-predicted categories)</h2><ul>{causes or '<li>No causes recorded</li>'}</ul>
<h2>Watch for: what went wrong in those incidents</h2><ul style="list-style:none;padding-left:4px">{watch_li or '<li>No causes recorded</li>'}</ul>
<h2>Fixes investigators recommended</h2><ul style="list-style:none;padding-left:4px">{fixes or '<li>No closely matching fix on record. Discuss controls at the pre-job meeting.</li>'}</ul>
<p class="foot">Checklist items are recommendations investigators wrote for past incidents (public BSEE reports).
A decision aid for supervisors, not a substitute for a job safety analysis.</p></body></html>"""
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
st.markdown(f"""<div class="hero"><h1>🔎 Incident Lens</h1>
<p>Learn from {len(inc):,} real offshore incident reports without reading every one. Find look-alike incidents, see likely causes
and why, and spot reports missing key details.</p>
<div style="margin-top:12px"><span class="chip">Track 9 · Local Business &amp; Industry</span>
<span class="chip">Real BSEE report text only (synthetic cells excluded)</span>
<span class="chip">Cause model macro-F1 {best['macro_f1']:.2f} vs baseline {base['macro_f1']:.2f}</span>
<span class="chip">Runs fully local · no data leaves the laptop</span></div></div>""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["🌌  Incident galaxy", "🔍  Similar incidents", "🧠  Cause & why",
                                        "⚠️  Report gaps & trends", "📊  How well it works"])

# ---------- TAB 0: galaxy ----------
with tab0:
    st.markdown("#### Every incident, placed by meaning")
    st.caption(f"Each dot is one of {len(inc):,} real reports. Reports that describe similar events sit close together. "
               "Colors are 12 themes the model discovered on its own (k-means on the embeddings; we named them by their top words). Hover a dot to read it.")
    st.plotly_chart(galaxy_fig(), use_container_width=True)
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
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB", xaxis_title="cause statements in look-alikes")
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
                       "only 373 of 1,212 reports contain an actionable fix at all).")
            for r in recs:
                st.markdown(f"<div class='lesson'>{esc(r[:260])}</div>", unsafe_allow_html=True)
            if not recs:
                st.caption("No recommendations recorded for these incidents.")
            st.markdown("**Where this report lands**")
            star = (float(res.x.mean()), float(res.y.mean()))
            st.plotly_chart(galaxy_fig(highlight=top, star=star, height=300, legend=False), use_container_width=True)
            tally_s = tally.to_dict() if all_causes else {}
            html_doc = alert_html(q, res, tally_s, recs, [c for cs in res.causes for c in cs][:6])
            if st.button("⚠️  Generate safety alert", type="primary", use_container_width=True):
                st.session_state["alert"] = html_doc
            if st.session_state.get("alert"):
                st.download_button("⬇  Download alert (HTML, printable)", st.session_state["alert"], "safety_alert.html",
                                   "text/html", use_container_width=True)
    if st.session_state.get("alert") and q.strip():
        st.markdown("#### Safety alert preview")
        import streamlit.components.v1 as components
        components.html(st.session_state["alert"], height=760, scrolling=True)
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
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB")
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
    for cname, colr in [("No cause recorded", "#FB7185"), ("No actionable fix", "#F59E0B")]:
        fig.add_trace(go.Scatter(x=yr.index, y=yr[cname], name=cname, mode="lines+markers", line_color=colr))
    fig.update_layout(height=280, title="Share of reports missing key details, by year (%)",
                      margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#E5E7EB", legend=dict(orientation="h"))
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
    k3.markdown(f"<div class='kpi'><div class='big' style='color:#F59E0B'>{best['macro_f1']:.2f}</div><div class='lbl'>Incident Lens (meaning embeddings)</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        names = [n for n in EV["results"] if EV["results"][n]["ci"]]
        fig = go.Figure()
        for n in names:
            r = EV["results"][n]
            fig.add_trace(go.Scatter(x=[r["macro_f1"]], y=[n], mode="markers", marker=dict(size=14, color="#F59E0B" if n == EV["best"] else "#64748B"),
                                     error_x=dict(type="data", symmetric=False, array=[r["ci"][1] - r["macro_f1"]], arrayminus=[r["macro_f1"] - r["ci"][0]]),
                                     showlegend=False))
        fig.update_layout(title="Macro-F1 with 95% interval (no overlap = real difference)", height=300,
                          margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#E5E7EB", xaxis_range=[0.2, 0.6])
        st.plotly_chart(fig, use_container_width=True)
    with b:
        labs = EV["labels"]
        fig = go.Figure([go.Bar(name="Baseline", x=labs, y=[base["per_class"][l] for l in labs], marker_color="#64748B"),
                         go.Bar(name="Incident Lens", x=labs, y=[best["per_class"][l] for l in labs], marker_color="#F59E0B")])
        fig.update_layout(title="F1 per category: biggest gains on the rare ones", barmode="group", height=330,
                          margin=dict(l=0, r=0, t=70, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#E5E7EB", legend=dict(orientation="h", y=1.12, x=0), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    with c:
        cm = np.array(EV["confusion"])
        fig = go.Figure(go.Heatmap(z=cm, x=labs, y=labs, colorscale="YlOrBr", text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Where it gets confused (rows = investigator, cols = model)", height=380,
                          margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB",
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
