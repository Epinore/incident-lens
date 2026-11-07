// Build "Team 2 slides.pptx" for Incident Lens. Run: node build_slides.js
// 8 slides for a 4-minute talk, mapped to the five judging criteria.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaDatabase, FaHistory, FaLayerGroup, FaUserShield, FaSearch, FaBrain, FaExclamationTriangle, FaBell,
  FaHardHat, FaClipboardCheck, FaLock, FaGithub, FaCheck, FaTimes, FaArrowRight,
} = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.join(__dirname, "..");
const SHOT = (f) => path.join(ROOT, "screenshots", f);
const ev = JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts", "eval.json")));
const gold = JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts", "gold.json")));
const R = ev.results;
const base = R["Baseline: TF-IDF + LR (official)"], best = R[ev.best], maj = R["Majority class (always guess most common)"];
const hyb = R["Hybrid: TF-IDF + embeddings + LR"];
const f2 = (x) => x.toFixed(2);
const pct = (x) => Math.round(100 * x);

// Deep-ocean palette (matches the app)
const NAVY = "082C44", PANEL = "0E3A55", EDGE = "1E5A70", TEAL = "2DD4BF", MINT = "5EEAD4", HOT = "FF7A45", SKY = "38BDF8";
const WHITE = "FFFFFF", SOFT = "D5EEEA", MUTED = "8FB3BF", DIM = "7FA7B5";
const THEME = {
  name: "Incident Lens", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "0B1B26", lt1: WHITE, dk2: NAVY, lt2: "E6F4F1", accent1: TEAL, accent2: SKY, accent3: "A78BFA", accent4: "FBBF24", accent5: HOT, accent6: "F472B6", hlink: MINT, folHlink: "A78BFA" },
};

async function icon(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Incident Lens";
  pres.author = "Ethan Leng and Oluchi Nnadi (Team 2)";

  const BG = { path: path.join(__dirname, "bg.png") };
  pres.defineSlideMaster({ title: "Cover", background: BG });
  pres.defineSlideMaster({
    title: "Ocean", background: BG,
    objects: [{ text: { text: "Incident Lens  ·  Team 2  ·  Track 9 (Local Business & Industry)", options: { x: 0.5, y: 5.28, w: 6, h: 0.22, fontSize: 8.5, color: DIM, fontFace: "Calibri" } } }],
    slideNumber: { x: 9.1, y: 5.28, w: 0.4, h: 0.22, fontSize: 8.5, color: DIM },
  });

  // ---------- helpers ----------
  const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontFace: "Calibri", color: SOFT, valign: "top" }, o));
  const eyebrow = (s, text, color = TEAL, y = 0.32) => T(s, text, { x: 0.5, y, w: 9, h: 0.25, fontSize: 10.5, bold: true, color, charSpacing: 2, fontFace: "Arial" });
  const title = (s, text, y = 0.55) => T(s, text, { x: 0.5, y, w: 9, h: 0.6, fontSize: 26, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
  const panel = (s, x, y, w, h, line = EDGE, fill = PANEL, tr = 22) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill, transparency: tr }, line: { color: line, width: 0.75 } });
  const badge = async (s, x, y, d, Icon, color = TEAL) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: NAVY }, line: { color, width: 1.25 } });
    s.addImage({ data: await icon(Icon, color), x: x + d * 0.27, y: y + d * 0.27, w: d * 0.46, h: d * 0.46 });
  };
  const num = (s, x, y, n, color = TEAL) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.34, h: 0.34, fill: { color }, line: { color } });
    T(s, String(n), { x, y, w: 0.34, h: 0.34, fontSize: 12, bold: true, color: NAVY, align: "center", valign: "middle", fontFace: "Arial" });
  };
  const arrow = (s, x, y, w) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: MINT, width: 1.5, endArrowType: "triangle" } });
  const vline = (s, x, y, h) => s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: MINT, width: 1.5 } });

  // =====================================================================
  // 1. Introduction + problem statement
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Cover", sectionTitle: "Introduction" });
  s.addImage({ path: path.join(__dirname, "galaxy_alpha.png"), x: 4.85, y: 0.2, w: 5.15, h: 3.07 });
  T(s, "TEAM 2  ·  TRACK 9 (LOCAL BUSINESS & INDUSTRY)", { x: 0.5, y: 0.5, w: 4.3, h: 0.25, fontSize: 9.5, bold: true, color: TEAL, charSpacing: 1, fontFace: "Arial" });
  T(s, "Incident Lens", { x: 0.5, y: 0.82, w: 5, h: 0.85, fontSize: 46, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
  T(s, "Learn from 1,212 offshore incident reports without reading every one", { x: 0.5, y: 1.72, w: 4.3, h: 0.7, fontSize: 17, italic: true, color: MINT, fontFace: "Georgia" });
  T(s, [{ text: "By: ", options: { color: DIM } }, { text: "Ethan Leng and Oluchi Nnadi", options: { color: WHITE, bold: true } }],
    { x: 0.5, y: 2.6, w: 4.3, h: 0.35, fontSize: 15, valign: "middle" });

  panel(s, 0.5, 3.45, 9, 1.55, EDGE, PANEL, 15);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.6, w: 0.06, h: 1.25, fill: { color: HOT }, line: { color: HOT } });
  T(s, "PROBLEM STATEMENT", { x: 0.8, y: 3.6, w: 8.4, h: 0.25, fontSize: 10, bold: true, color: HOT, charSpacing: 2, fontFace: "Arial" });
  T(s, "Safety teams want to learn from past incidents, but finding relevant cases in hundreds of long reports takes time. Incident Lens helps them find similar incidents and see what investigators recommended.",
    { x: 0.8, y: 3.92, w: 8.4, h: 0.95, fontSize: 15, color: WHITE });
  s.addNotes("Hi, we're Ethan and Oluchi, Team 2, Track 9. Offshore safety teams file hundreds of incident reports, but the lessons stay buried in free text. Only about a quarter of the reports even write down a fix someone could act on. Before a risky job nobody can read 1,212 reports, so the same incidents happen again. Incident Lens lets a safety team learn from all of them in a minute.");

  // =====================================================================
  // 2. Process 1: understand the data
  pres.addSection({ title: "Process" });
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Process" });
  eyebrow(s, "HOW WE BUILT IT  ·  STEP 1 OF 3");
  title(s, "How we prepared the data");
  const rules = [
    [FaDatabase, "Real text only", "Used real report text, not generated data."],
    [FaHistory, "One time period", "We used 2019+ data to keep the category definitions consistent."],
    [FaLayerGroup, "Split by incident", "Kept each incident entirely in training or testing within each round."],
    [FaUserShield, "No names", "Excluded name fields. Masked people, company and rig names in displayed text."],
  ];
  for (let i = 0; i < 4; i++) {
    const [I, h, d] = rules[i];
    const x = 0.5 + i * 2.27;
    panel(s, x, 1.45, 2.1, 3.0);
    await badge(s, x + 0.2, 1.65, 0.62, I);
    T(s, h, { x: x + 0.2, y: 2.42, w: 1.75, h: 0.4, fontSize: 15, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
    T(s, d, { x: x + 0.2, y: 2.88, w: 1.75, h: 1.5, fontSize: 12.5, color: SOFT });
  }
  T(s, "Data: public US federal offshore incident reports (BSEE), Energy Institute partner challenge dataset, real_only files.",
    { x: 0.5, y: 4.62, w: 9, h: 0.3, fontSize: 10.5, italic: true, color: DIM });
  s.addNotes("Step one was reading the data guide before modelling. Some cells are made up, so we used real text only. Cause names changed over the years, so we compare within one period, 2019 on. We split tests by incident so the model never sees the same incident twice. And names never appear in the tool.");

  // =====================================================================
  // 3. Process 2: one pipeline, three tools
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Process" });
  eyebrow(s, "HOW WE BUILT IT  ·  STEP 2 OF 3");
  title(s, "How Incident Lens works");
  // source
  panel(s, 0.5, 2.0, 1.55, 1.3);
  T(s, "1,212", { x: 0.5, y: 2.1, w: 1.55, h: 0.55, fontSize: 26, bold: true, color: WHITE, align: "center", fontFace: "Arial", valign: "middle" });
  T(s, "real incident reports", { x: 0.6, y: 2.65, w: 1.35, h: 0.55, fontSize: 11, color: SOFT, align: "center" });
  arrow(s, 2.08, 2.65, 0.32);
  // embeddings
  panel(s, 2.43, 1.85, 1.85, 1.6, TEAL);
  await badge(s, 3.08, 1.97, 0.55, FaBrain);
  T(s, "Meaning embeddings", { x: 2.45, y: 2.58, w: 1.81, h: 0.35, fontSize: 11.5, bold: true, color: WHITE, align: "center", fontFace: "Arial", valign: "middle" });
  T(s, "open source, runs on a computer, bge-small", { x: 2.55, y: 2.93, w: 1.61, h: 0.45, fontSize: 10, color: SOFT, align: "center" });
  arrow(s, 4.3, 2.65, 0.2);
  vline(s, 4.5, 1.73, 1.84);
  const tools = [
    [FaSearch, "Find similar incidents", "See past cases and recommended actions"],
    [FaBrain, "Classify causes", "Suggest a category and highlight influential words"],
    [FaExclamationTriangle, "Check missing details", "Flag missing causes, actions or dates"],
  ];
  for (let i = 0; i < 3; i++) {
    const [I, h, d] = tools[i];
    const y = 1.35 + i * 0.92;
    arrow(s, 4.5, y + 0.38, 0.25);
    panel(s, 4.78, y, 2.75, 0.76);
    await badge(s, 4.9, y + 0.15, 0.46, I);
    T(s, h, { x: 5.45, y: y + 0.06, w: 2.0, h: 0.28, fontSize: 12, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
    T(s, d, { x: 5.45, y: y + 0.34, w: 2.02, h: 0.4, fontSize: 9.5, color: SOFT });
    s.addShape(pres.shapes.LINE, { x: 7.53, y: y + 0.38, w: 0.22, h: 0, line: { color: MINT, width: 1.5 } });
  }
  vline(s, 7.75, 1.73, 1.84);
  arrow(s, 7.75, 2.65, 0.25);
  panel(s, 8.03, 1.85, 1.47, 1.6, HOT, "3A2A2A", 30);
  await badge(s, 8.49, 1.97, 0.55, FaBell, HOT);
  T(s, "Safety alert", { x: 8.06, y: 2.58, w: 1.41, h: 0.35, fontSize: 12.5, bold: true, color: WHITE, align: "center", fontFace: "Arial", valign: "middle" });
  T(s, "Print relevant cases and lessons for the team", { x: 8.1, y: 2.93, w: 1.33, h: 0.5, fontSize: 9.5, color: SOFT, align: "center" });
  // challenge fit
  panel(s, 0.5, 4.2, 9, 0.75, EDGE, NAVY, 30);
  T(s, [
    { text: "The challenge asked for one of these three. ", options: { color: SOFT } },
    { text: "We built all three. ", options: { color: WHITE, bold: true } },
    { text: "We checked the model against existing labels and our own hand-labeled examples.", options: { color: SOFT } },
  ], { x: 0.75, y: 4.25, w: 8.5, h: 0.65, fontSize: 12.5, valign: "middle" });
  s.addNotes("Step two: one pipeline. Every report becomes a meaning vector with a small open-source model that runs on a laptop. From that we built all three tools the challenge lists: similar-incident search, cause patterns, and flags for missing details. They all feed one output: a printable safety alert.");

  // =====================================================================
  // 4. Process 3: test and iterate
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Process" });
  eyebrow(s, "HOW WE BUILT IT  ·  STEP 3 OF 3");
  title(s, "We tested three models and kept the one that won");
  panel(s, 0.5, 1.35, 5.6, 3.6);
  T(s, "Macro-F1 on cause category (higher is better)", { x: 0.7, y: 1.47, w: 5.2, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
  const bars = [["Incident Lens: meaning", best.macro_f1, TEAL], ["Hybrid: words + meaning", hyb.macro_f1, SKY], ["Official baseline: word counts", base.macro_f1, "5B7D8E"], ["Always guess the most common", maj.macro_f1, "3E5A6B"]];
  bars.forEach(([lab, v, c], i) => {
    const y = 1.95 + i * 0.72, x0 = 2.75, wmax = 2.6;
    T(s, lab, { x: 0.65, y, w: 2.0, h: 0.5, fontSize: 11, color: SOFT, align: "right", valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y: y + 0.05, w: Math.max(0.05, wmax * v / 0.5), h: 0.4, fill: { color: c }, line: { color: c } });
    T(s, f2(v), { x: x0 + wmax * v / 0.5 + 0.08, y, w: 0.6, h: 0.5, fontSize: 13, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
  });
  panel(s, 6.3, 1.35, 3.2, 3.6, TEAL);
  T(s, f2(best.macro_f1), { x: 6.5, y: 1.45, w: 2.8, h: 0.8, fontSize: 44, bold: true, color: TEAL, fontFace: "Arial", valign: "middle" });
  T(s, `vs ${f2(base.macro_f1)} for the official baseline`, { x: 6.5, y: 2.22, w: 2.8, h: 0.3, fontSize: 13, color: WHITE, bold: true });
  T(s, [
    { text: `Same ${ev.n} rows, same folds, same metric as the baseline.`, options: { bullet: { indent: 12 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: `95% intervals don't overlap: ${f2(base.ci[0])}–${f2(base.ci[1])} vs ${f2(best.ci[0])}–${f2(best.ci[1])}.`, options: { bullet: { indent: 12 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: `Rare causes gain most: Communication ${f2(base.per_class["Communication"])} → ${f2(best.per_class["Communication"])}.`, options: { bullet: { indent: 12 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "Adding word counts back made it worse, so we kept the simpler model.", options: { bullet: { indent: 12 } } },
  ], { x: 6.5, y: 2.65, w: 2.85, h: 2.2, fontSize: 11, color: SOFT });

  s.addNotes(`Step three: we tested honestly. Same ${ev.n} rows, same folds and metric as the official baseline. Word counts score ${f2(base.macro_f1)}. Meaning embeddings score ${f2(best.macro_f1)}, and the confidence intervals don't overlap. We also tried mixing both; it was worse, so we kept the simpler model.`);

  // =====================================================================
  // 5. Demo
  pres.addSection({ title: "Demo" });
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Demo" });
  eyebrow(s, "LIVE DEMO", HOT);
  title(s, "Find lessons from a past incident");
  const steps = [
    ["Enter an incident", ""],
    ["Review similar cases", "See recorded causes and recommendations."],
    ["Ask why", "Cause category plus the words that drove it"],
    ["Print the safety alert", "A one-page briefing for the pre-job meeting"],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.4 + i * 0.86;
    num(s, 0.5, y + 0.03, i + 1, i === 3 ? HOT : TEAL);
    T(s, h, { x: 0.98, y, w: 2.7, h: 0.32, fontSize: 13, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
    T(s, d, { x: 0.98, y: y + 0.33, w: 2.7, h: 0.42, fontSize: 10.5, color: SOFT });
  });
  s.addImage({ path: SHOT("1b-new-report-lands.jpg"), x: 3.85, y: 1.35, w: 3.85, h: 2.09 });
  s.addImage({ path: SHOT("1-similar-incidents.jpg"), x: 3.85, y: 3.55, w: 1.875, h: 1.4, sizing: { type: "cover", w: 1.875, h: 1.4 } });
  s.addImage({ path: SHOT("2-cause-and-why.jpg"), x: 5.825, y: 3.55, w: 1.875, h: 1.4, sizing: { type: "cover", w: 1.875, h: 1.4 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.95, y: 1.35, w: 1.5, h: 1.5, rectRadius: 0.08, fill: { color: WHITE }, line: { color: TEAL, width: 1 } });
  s.addImage({ path: path.join(__dirname, "qr_app.png"), x: 8.01, y: 1.41, w: 1.38, h: 1.38, hyperlink: { url: "https://incident-lens.streamlit.app" } });
  T(s, [{ text: "Try it live:", options: { breakLine: true } }, { text: "incident-lens.streamlit.app", options: { color: MINT, bold: true, breakLine: true, hyperlink: { url: "https://incident-lens.streamlit.app" } } }, { text: "Screenshots are the backup." }],
    { x: 7.85, y: 3.0, w: 1.7, h: 0.9, fontSize: 9.5, italic: true, color: DIM });
  s.addNotes("Live demo. Galaxy: 1,212 incidents, 12 themes. Similar incidents: paste the crane sling report; it lands next to past crane incidents, with causes and fixes. Cause & why: highlighted words. Then one click: the printable safety alert. If the demo fails, use these screenshots.");

  // =====================================================================
  // 6. AI we used and how we checked it (required title)
  pres.addSection({ title: "AI check" });
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "AI check" });
  eyebrow(s, "REQUIRED SLIDE");
  title(s, "AI we used and how we checked it");
  const tiles = [
    [f2(best.macro_f1), `macro-F1 vs ${f2(base.macro_f1)} official baseline (${ev.n} rows, 5 folds grouped by incident)`, TEAL],
    [`${gold.agree} of ${gold.n}`, `our hand labels agree with the model, on incidents it never saw (${pct(gold.agree_rate)}%, CI ${pct(gold.ci95[0])}–${pct(gold.ci95[1])}%)`, SKY],
    [`${gold.team_vs_investigator.agree} of ${gold.team_vs_investigator.n}`, "times we matched the original investigators. People disagree on causes too.", HOT],
  ];
  tiles.forEach(([n, l, c], i) => {
    const x = 0.5 + i * 3.05;
    panel(s, x, 1.3, 2.9, 1.15, c);
    T(s, n, { x: x + 0.18, y: 1.36, w: 2.6, h: 0.52, fontSize: 26, bold: true, color: c, fontFace: "Arial", valign: "middle" });
    T(s, l, { x: x + 0.18, y: 1.88, w: 2.6, h: 0.52, fontSize: 9.5, color: SOFT });
  });
  const bl = (head, items) => [{ text: head, options: { bold: true, fontSize: 13.5, color: WHITE, breakLine: true, paraSpaceAfter: 6, fontFace: "Arial" } }]
    .concat(items.map((t, i) => ({ text: t, options: { bullet: { indent: 12 }, fontSize: 11, color: SOFT, breakLine: i < items.length - 1, paraSpaceAfter: 4 } })));
  panel(s, 0.5, 2.6, 4.42, 2.38);
  T(s, bl("AI inside the product", [
    "bge-small text embeddings (open source, local): similar-incident search",
    "Logistic regression on embeddings: cause category (6 types)",
    "k-means on embeddings: 12 incident themes",
    "Leave-one-word-out: highlights why it chose a cause",
    "Claude & Codex helped us write code; it is not in the product",
  ]), { x: 0.7, y: 2.72, w: 4.05, h: 2.2 });
  panel(s, 5.08, 2.6, 4.42, 2.38);
  T(s, bl("How we checked it", [
    "Compared with the official baseline and a guess-the-most-common rule, 95% bootstrap intervals",
    `We hand-labeled ${gold.n} causes from ${gold.n_incidents} incidents; used only to test, never to tune`,
    "Each labeled row is scored by a model that never saw that incident",
    "Most common mistake: Human performance read as Equipment failure (18) or Supervision (16)",
  ]), { x: 5.28, y: 2.72, w: 4.05, h: 2.2 });
  s.addNotes(`The AI is inside the product: embeddings for search, a classifier for causes, k-means for themes, and word highlights for why. We checked it three ways: against the official baseline, ${f2(best.macro_f1)} vs ${f2(base.macro_f1)}; against ${gold.n} causes we labeled by hand, where it agrees about half the time on incidents it never saw; and we checked ourselves: we matched the investigators ${gold.team_vs_investigator.agree} of ${gold.team_vs_investigator.n} times.`);

  // =====================================================================
  // 7. Retrospective
  pres.addSection({ title: "Close" });
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Close" });
  eyebrow(s, "LOOKING BACK");
  title(s, "What worked, what didn't, and what's next");
  const cols = [
    [FaCheck, "What worked", TEAL, [
      `Meaning beats word counts: +${f2(best.macro_f1 - base.macro_f1)} macro-F1`,
      "Search + alert answer the real question: has this happened before?",
      "Reading the data guide first kept test incidents out of training",
    ]],
    [FaTimes, "What didn't (yet)", HOT, [
      "Supervision is hard: 28 training examples; we got 0 of 2 right",
      "Human-error causes often filed as Management systems",
      `Only ${gold.n} hand labels, so the interval is wide (${pct(gold.ci95[0])}–${pct(gold.ci95[1])}%)`,
      "Year trends not verified, so we don't claim them",
    ]],
    [FaArrowRight, "Next", SKY, [
      "Label more rows together with investigators",
      "Load one company's own reports",
      "Let users correct a cause and retrain the model",
    ]],
  ];
  for (let i = 0; i < 3; i++) {
    const [I, h, c, items] = cols[i];
    const x = 0.5 + i * 3.05;
    panel(s, x, 1.35, 2.9, 2.85, c);
    await badge(s, x + 0.2, 1.52, 0.5, I, c);
    T(s, h, { x: x + 0.82, y: 1.52, w: 1.95, h: 0.5, fontSize: 15, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
    T(s, items.map((t, j) => ({ text: t, options: { bullet: { indent: 12 }, breakLine: j < items.length - 1, paraSpaceAfter: 7 } })),
      { x: x + 0.2, y: 2.2, w: 2.55, h: 1.95, fontSize: 11.5, color: SOFT });
  }
  panel(s, 0.5, 4.35, 9, 0.62, EDGE, NAVY, 30);
  T(s, [{ text: "Biggest lesson: ", options: { bold: true, color: MINT } }, { text: "an honest 0.48 we can explain is worth more than a high score we can't trust.", options: { color: WHITE } }],
    { x: 0.75, y: 4.38, w: 8.5, h: 0.56, fontSize: 13, valign: "middle" });
  s.addNotes("Looking back. What worked: meaning beats word counts, and search plus the alert answers the question a crew actually asks. What didn't: Supervision is hard with only 28 examples, and human-error causes get filed as management systems. 59 labels is small, so the interval is wide, and we don't claim year trends we couldn't verify. Next: more labels with investigators, and a company's own reports.");

  // =====================================================================
  // 8. Real impact
  s = pres.addSlide({ masterName: "Ocean", sectionTitle: "Close" });
  eyebrow(s, "REAL IMPACT", HOT);
  title(s, "From buried reports to a one-minute pre-job briefing");
  const imp = [
    [FaHardHat, "Before the job", "A supervisor types tomorrow's task and gets past incidents, causes and fixes in under a minute, then prints the alert for the crew's pre-job talk."],
    [FaSearch, "During an investigation", "Investigators see look-alike cases and a suggested cause category, with the words behind it, instead of searching by keyword."],
    [FaClipboardCheck, "Better reports", "891 of 1,212 reports never wrote down a fix anyone could act on. The tool flags these gaps so the next report teaches more."],
  ];
  for (let i = 0; i < 3; i++) {
    const [I, h, d] = imp[i];
    const y = 1.35 + i * 1.2;
    await badge(s, 0.5, y + 0.1, 0.62, I, i === 2 ? HOT : TEAL);
    T(s, h, { x: 1.3, y, w: 4.7, h: 0.35, fontSize: 14.5, bold: true, color: WHITE, fontFace: "Arial", valign: "middle" });
    T(s, d, { x: 1.3, y: y + 0.36, w: 4.7, h: 0.78, fontSize: 11.5, color: SOFT });
  }
  panel(s, 6.3, 1.35, 3.2, 3.6, TEAL);
  T(s, "Ready to use tomorrow", { x: 6.5, y: 1.5, w: 2.85, h: 0.4, fontSize: 15, bold: true, color: TEAL, fontFace: "Arial", valign: "middle" });
  const ready = [
    [FaLock, "Runs on one laptop, no API"],
    [FaUserShield, "No people or companies named"],
    [FaGithub, "Free and open source"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.98 + i * 0.42;
    s.addImage({ data: await icon(ready[i][0], MINT), x: 6.52, y: y + 0.05, w: 0.22, h: 0.22 });
    T(s, ready[i][1], { x: 6.9, y, w: 2.5, h: 0.32, fontSize: 11.5, color: SOFT, valign: "middle" });
  }
  const REPO = "https://github.com/Epinore/incident-lens", APP = "https://incident-lens.streamlit.app";
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.5, y: 3.3, w: 1.36, h: 1.36, rectRadius: 0.08, fill: { color: WHITE }, line: { color: TEAL, width: 1 } });
  s.addImage({ path: path.join(__dirname, "qr_app.png"), x: 6.55, y: 3.35, w: 1.26, h: 1.26, hyperlink: { url: APP } });
  T(s, "Scan to try it live on your phone", { x: 8.0, y: 3.3, w: 1.4, h: 0.6, fontSize: 11, bold: true, color: WHITE });
  T(s, "Thank you!", { x: 8.0, y: 4.1, w: 1.45, h: 0.45, fontSize: 17, bold: true, color: TEAL, fontFace: "Arial", valign: "middle" });
  T(s, "incident-lens.streamlit.app", { x: 6.5, y: 4.68, w: 1.6, h: 0.22, fontSize: 9.5, color: MINT, hyperlink: { url: APP } });
  T(s, "Code: GitHub", { x: 8.1, y: 4.68, w: 1.3, h: 0.22, fontSize: 9.5, color: DIM, align: "right", hyperlink: { url: REPO } });
  s.addNotes("The real impact: before a job, a supervisor gets the lessons from every similar past incident in a minute and prints them for the crew. Investigators get look-alike cases and suggested causes. And because 891 reports never wrote down a usable fix, the tool shows where reporting itself can improve. It runs on one laptop, shows no names, and is open source, so a safety team could use it tomorrow. Thank you.");

  const out = path.join(ROOT, "slides", "Team 2 slides.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
