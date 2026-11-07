// Build "Team 2 slides.pptx" for Incident Lens. Run: node build_slides.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaSearch, FaBrain, FaExclamationTriangle, FaChartBar, FaLock, FaUserShield, FaHistory, FaBalanceScale, FaFileAlt, FaHardHat } = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.join(__dirname, "..");
const ev = JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts", "eval.json")));
const gold = fs.existsSync(path.join(ROOT, "artifacts", "gold.json")) ? JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts", "gold.json"))) : { n: 0 };
const R = ev.results;
const base = R["Baseline: TF-IDF + LR (official)"], best = R[ev.best], maj = R["Majority class (always guess most common)"];
const f2 = (x) => x.toFixed(2);

const NAVY = "0B1220", PANEL = "131C2E", AMBER = "F59E0B", SLATE = "64748B", WHITE = "FFFFFF", INK = "1F2937", MUTED = "9CA3AF", LIGHT = "F3F4F6", SKY = "38BDF8";
const THEME = {
  name: "Incident Lens", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "111827", lt1: "FFFFFF", dk2: NAVY, lt2: LIGHT, accent1: AMBER, accent2: SKY, accent3: "A78BFA", accent4: "34D399", accent5: "FB7185", accent6: SLATE, hlink: "2563EB", folHlink: "7C3AED" },
};

async function icon(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Incident Lens";
  pres.author = "Team 2";

  pres.defineSlideMaster({
    title: "Content",
    background: { color: WHITE },
    objects: [{ text: { text: "Incident Lens · Team 2 · Track 9 (Local Business & Industry)", options: { x: 0.5, y: 5.25, w: 6, h: 0.25, fontSize: 9, color: MUTED, fontFace: "Calibri" } } }],
    placeholders: [],
    slideNumber: { x: 9.1, y: 5.25, w: 0.4, h: 0.25, fontSize: 9, color: MUTED },
  });
  pres.defineSlideMaster({ title: "Dark", background: { color: NAVY } });

  const title = (s, t, dark = false) => s.addText(t, { x: 0.5, y: 0.3, w: 9, h: 0.7, fontSize: 30, bold: true, color: dark ? WHITE : INK, fontFace: "Arial", isTextBox: true, margin: 0 });

  // 1. Title
  pres.addSection({ title: "Intro" });
  let s = pres.addSlide({ masterName: "Dark", sectionTitle: "Intro" });
  s.addImage({ data: await icon(FaHardHat, AMBER), x: 0.5, y: 0.9, w: 0.7, h: 0.7 });
  s.addText("Incident Lens", { x: 0.5, y: 1.75, w: 9, h: 1.0, fontSize: 48, bold: true, color: WHITE, fontFace: "Arial", isTextBox: true, margin: 0 });
  s.addText("Learn from 1,200 offshore incident reports without reading every one", { x: 0.5, y: 2.75, w: 8.5, h: 0.6, fontSize: 20, color: AMBER, italic: true, isTextBox: true, margin: 0 });
  s.addText("Team 2  ·  Track 9 (Local Business & Industry)  ·  Energy Institute partner challenge", { x: 0.5, y: 4.4, w: 9, h: 0.4, fontSize: 14, color: MUTED, isTextBox: true, margin: 0 });
  s.addNotes("Hi, we're Team 2. Incident Lens helps offshore safety teams learn from past incidents without reading every report.");

  // 2. Problem
  pres.addSection({ title: "Problem" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
  title(s, "The lessons are buried in free text");
  const stats = [["1,212", "real offshore incident reports (US federal BSEE data)"], ["3,572", "cause statements written by investigators"], ["1,230", "recommended fixes, scattered across separate reports"]];
  stats.forEach(([n, l], i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.35, w: 2.8, h: 1.9, fill: { color: LIGHT }, rectRadius: 0.12, line: { color: LIGHT } });
    s.addText(n, { x: x + 0.2, y: 1.5, w: 2.4, h: 0.9, fontSize: 44, bold: true, color: i === 2 ? AMBER : INK, fontFace: "Arial", isTextBox: true, margin: 0 });
    s.addText(l, { x: x + 0.2, y: 2.4, w: 2.4, h: 0.7, fontSize: 14, color: "4B5563", isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText([
    { text: "Who it's for: ", options: { bold: true } },
    { text: "offshore safety teams and incident investigators. When a new incident happens, they need to know: has this happened before, what caused it, and what fix worked?" },
  ], { x: 0.5, y: 3.6, w: 9, h: 1.0, fontSize: 16, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Offshore teams file hundreds of incident reports. The lessons sit in free text, so nobody reads them all and the same incidents happen again.");

  // 3. What we built
  pres.addSection({ title: "Product" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Product" });
  title(s, "What we built: one tool, four views");
  const feats = [
    [FaSearch, "Similar incidents", "Paste a new report and get the closest past incidents, their causes and the fixes that were recommended."],
    [FaBrain, "Cause & why", "Predicts the investigator's cause category and highlights the words that drove the answer."],
    [FaExclamationTriangle, "Report gaps", "Flags reports missing a cause, a recommendation, an outcome or a date."],
    [FaChartBar, "How well it works", "Honest scores against the official baseline, with intervals, errors and limits."],
  ];
  for (let i = 0; i < 4; i++) {
    const [I, h, d] = feats[i];
    const x = 0.5 + (i % 2) * 4.6, y = 1.25 + Math.floor(i / 2) * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: 1.75, fill: { color: LIGHT }, rectRadius: 0.12, line: { color: LIGHT } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: NAVY }, line: { color: NAVY } });
    s.addImage({ data: await icon(I, AMBER), x: x + 0.42, y: y + 0.47, w: 0.36, h: 0.36 });
    s.addText(h, { x: x + 1.15, y: y + 0.25, w: 3.1, h: 0.4, fontSize: 18, bold: true, color: INK, fontFace: "Arial", isTextBox: true, margin: 0 });
    s.addText(d, { x: x + 1.15, y: y + 0.68, w: 3.1, h: 0.95, fontSize: 14, color: "4B5563", isTextBox: true, margin: 0, valign: "top" });
  }
  s.addNotes("Four views. Now a live demo. (Switch to the app: paste the crane example, show look-alikes and fixes; then Cause & why with highlighted words.)");

  // 4. Demo screenshots (backup)
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "Product" });
  title(s, "Live demo (screenshots as backup)", true);
  s.addImage({ path: path.join(ROOT, "screenshots", "1-similar-incidents.jpg"), x: 0.5, y: 1.2, w: 4.4, h: 2.75 });
  s.addImage({ path: path.join(ROOT, "screenshots", "2-cause-and-why.jpg"), x: 5.1, y: 1.2, w: 4.4, h: 2.75 });
  s.addText("A dropped-load report finds a 70% match: an earlier crane incident, plus the fixes recommended then", { x: 0.5, y: 4.1, w: 4.4, h: 0.7, fontSize: 13, color: LIGHT, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Cause & why: the category, how sure the model is, which words mattered, and labeled precedents", { x: 5.1, y: 4.1, w: 4.4, h: 0.7, fontSize: 13, color: LIGHT, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Only if the live demo fails: walk through these two screenshots.");

  // 5. Results
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  title(s, `Cause model: macro-F1 ${f2(best.macro_f1)} vs baseline ${f2(base.macro_f1)}`);
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels: ["Always guess most common", "Official baseline (word counts)", "Incident Lens (meaning)"], values: [maj.macro_f1, base.macro_f1, best.macro_f1] }], {
    x: 0.4, y: 1.15, w: 4.5, h: 3.2, barDir: "bar", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd",
    chartColors: ["CBD5E1", SLATE, AMBER], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: "4B5563", catAxisLabelFontSize: 11, dataLabelFontSize: 12, dataLabelColor: INK, catAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    showTitle: true, title: "Macro-F1 (higher is better)", titleFontSize: 12, titleColor: INK, titleFontFace: "+mn-lt", valAxisMaxVal: 0.6,
  });
  const labs = ev.labels;
  s.addChart(pres.charts.BAR, [
    { name: "Baseline", labels: labs, values: labs.map((l) => base.per_class[l]) },
    { name: "Incident Lens", labels: labs, values: labs.map((l) => best.per_class[l]) },
  ], {
    x: 5.0, y: 1.15, w: 4.6, h: 3.2, barDir: "col", barGrouping: "clustered", chartColors: [SLATE, AMBER], showLegend: true, legendPos: "t",
    legendFontSize: 11, legendFontFace: "+mn-lt", valAxisLabelColor: "4B5563", catAxisLabelColor: "4B5563", catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, valAxisLabelFormatCode: "0.0",
    showTitle: true, title: "F1 per cause category", titleFontSize: 12, titleColor: INK, titleFontFace: "+mn-lt",
  });
  s.addText(`95% intervals don't overlap: baseline ${f2(base.ci[0])}–${f2(base.ci[1])}, ours ${f2(best.ci[0])}–${f2(best.ci[1])}. The rare categories gain most (Work environment ${f2(base.per_class["Work environment"])} → ${f2(best.per_class["Work environment"])}, Communication ${f2(base.per_class["Communication"])} → ${f2(best.per_class["Communication"])}).`,
    { x: 0.5, y: 4.4, w: 9, h: 0.7, fontSize: 13, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Same 374 rows, same folds, same metric as the official baseline. Macro-F1 averages over all six categories, so rare ones count as much as common ones. The baseline counts words; we use meaning embeddings, which tell communication from supervision far better.");

  // 6. AI we used and how we checked it  (required title)
  pres.addSection({ title: "AI check" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "AI check" });
  title(s, "AI we used and how we checked it");
  const goldLine = gold.n > 0
    ? `Our team hand-labeled ${gold.n} held-out rows. The model agreed on ${gold.agree} of ${gold.n} (${Math.round(100 * gold.agree / gold.n)}%).`
    : "Our team hand-labels a sample of the gold rows; the app compares them with the model.";
  const colL = [
    ["AI inside the product", true],
    ["bge-small text embeddings (open source, runs locally) for similar-incident search", false],
    ["Logistic regression on those embeddings to predict the cause category", false],
    ["Leave-one-word-out highlights to show why", false],
    ["Claude (AI coding assistant) helped write the code; it is not part of the product", false],
  ];
  const colR = [
    ["How we checked it", true],
    [`${ev.n} cause statements, 2019+ only, 5 folds grouped by incident so no incident is in both train and test`, false],
    [`Compared with the official baseline and an always-guess-most-common rule, with 95% bootstrap intervals`, false],
    [goldLine, false],
    ["We read its mistakes: it most often calls Human performance causes Equipment failure (18) or Supervision (16)", false],
  ];
  const bl = (arr) => arr.map(([t, h], i) => ({ text: t, options: h ? { bold: true, fontSize: 17, color: INK, breakLine: true, paraSpaceAfter: 6 } : { bullet: true, fontSize: 14, color: "374151", breakLine: i < arr.length - 1, paraSpaceAfter: 6 } }));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.2, w: 4.4, h: 3.85, fill: { color: LIGHT }, rectRadius: 0.12, line: { color: LIGHT } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.1, y: 1.2, w: 4.4, h: 3.85, fill: { color: "FEF3C7" }, rectRadius: 0.12, line: { color: "FEF3C7" } });
  s.addText(bl(colL), { x: 0.7, y: 1.35, w: 4.0, h: 3.6, isTextBox: true, margin: 0, valign: "top" });
  s.addText(bl(colR), { x: 5.3, y: 1.35, w: 4.0, h: 3.6, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("This is the slide the rules ask for. Left: the AI in the product. Right: how we checked it, including our own hand labels.");

  // 7. Responsible use + next
  pres.addSection({ title: "Close" });
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "Close" });
  title(s, "Responsible by design, ready to pilot", true);
  const resp = [
    [FaLock, "Runs fully local", "No API, no data leaves the laptop"],
    [FaUserShield, "No names", "Name fields never shown; obvious names scrubbed"],
    [FaFileAlt, "Real data only", "Synthetic cells excluded from training and scores"],
    [FaBalanceScale, "Honest limits", "Matches the investigator's label, not the truth; rare classes have about 27 examples each"],
  ];
  for (let i = 0; i < 4; i++) {
    const [I, h, d] = resp[i];
    const y = 1.2 + i * 0.85;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.6, h: 0.6, fill: { color: PANEL }, line: { color: AMBER, width: 1 } });
    s.addImage({ data: await icon(I, AMBER), x: 0.65, y: y + 0.15, w: 0.3, h: 0.3 });
    s.addText([{ text: h + "  ", options: { bold: true, color: WHITE } }, { text: d, options: { color: "CBD5E1" } }], { x: 1.3, y, w: 4.4, h: 0.6, fontSize: 14, isTextBox: true, margin: 0, valign: "middle" });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.0, y: 1.2, w: 3.5, h: 3.25, fill: { color: PANEL }, rectRadius: 0.12, line: { color: PANEL } });
  s.addText([
    { text: "Next steps", options: { bold: true, fontSize: 18, color: AMBER, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Load a company's own incident reports", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Investigators confirm or correct labels, and the model retrains", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Weekly digest: new incidents that look like past serious ones", options: { bullet: true } },
  ], { x: 6.2, y: 1.35, w: 3.1, h: 3.0, fontSize: 14, color: "E5E7EB", isTextBox: true, margin: 0, valign: "top" });
  s.addText("github.com/Epinore/incident-lens", { x: 0.5, y: 4.85, w: 9, h: 0.4, fontSize: 14, color: AMBER, isTextBox: true, margin: 0, hyperlink: { url: "https://github.com/Epinore/incident-lens" } });
  s.addNotes("It runs locally, shows no names and uses real data only. Next: a company's own reports, with investigators correcting labels. Thank you.");

  const out = path.join(ROOT, "slides", "Team 2 slides.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
