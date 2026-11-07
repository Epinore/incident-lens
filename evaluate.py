"""Cause-category classifier: compare against the official TF-IDF baseline (macro-F1 0.349).

Same rows (cause_task.csv, regime modern_six, n=374), same grouped folds (syn_fold),
same metric (macro-F1, 95% bootstrap over incidents). Writes artifacts/eval.json,
artifacts/oof.csv and the final models used by the app.
"""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

TASK = Path(r"D:\Projects\psm-incident-ml\data\processed\e19\cause_task.csv")
OUT = Path(__file__).parent / "artifacts"
NAMES = {"3": "Human performance", "15": "Equipment failure", "8": "Management systems",
         "17": "Supervision", "6": "Work environment", "9": "Communication"}
SEED, NBOOT = 0, 1000


def embed(texts):
    from fastembed import TextEmbedding
    m = TextEmbedding("BAAI/bge-small-en-v1.5")
    e = np.array(list(m.embed(texts, batch_size=32)), dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2)


def lr(C=1.0):
    return LogisticRegression(class_weight="balanced", max_iter=3000, C=C, random_state=SEED)


def oof(df, X_fn):
    pred = np.empty(len(df), dtype=object)
    for f in sorted(df.syn_fold.unique()):
        tr, te = (df.syn_fold != f).values, (df.syn_fold == f).values
        Xtr, Xte = X_fn(tr, te)
        m = lr(C=X_fn.C).fit(Xtr, df.y[tr])
        pred[te] = m.predict(Xte)
    return pred


def boot_ci(df, pred, labels):
    rng = np.random.default_rng(SEED)
    groups = df["Incident Number"].values
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    y = df.y.values
    s = []
    for _ in range(NBOOT):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        s.append(f1_score(y[pick], pred[pick], labels=labels, average="macro", zero_division=0))
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    df = pd.read_csv(TASK, dtype={"xw_element": str})
    df = df[df.syn_regime == "modern_six"].reset_index(drop=True)
    df["y"] = df.xw_element.astype(str)
    text = df.src_text_stripped.fillna("").tolist()
    labels = sorted(df.y.unique(), key=int)
    E = embed(text)

    def f_tfidf(tr, te):
        v = tfidf().fit([text[i] for i in np.where(tr)[0]])
        return v.transform([text[i] for i in np.where(tr)[0]]), v.transform([text[i] for i in np.where(te)[0]])
    f_tfidf.C = 1.0

    def f_emb(tr, te):
        return E[tr], E[te]
    f_emb.C = 4.0

    def f_both(tr, te):
        a, b = f_tfidf(tr, te)
        return hstack([a, csr_matrix(E[tr])]).tocsr(), hstack([b, csr_matrix(E[te])]).tocsr()
    f_both.C = 2.0

    results, preds = {}, {}
    maj = df.y.value_counts().idxmax()
    for name, fn in [("Baseline: TF-IDF + LR (official)", f_tfidf),
                     ("Embeddings + LR", f_emb),
                     ("Hybrid: TF-IDF + embeddings + LR", f_both)]:
        p = oof(df, fn)
        preds[name] = p
        per = f1_score(df.y, p, labels=labels, average=None, zero_division=0)
        results[name] = {
            "macro_f1": float(f1_score(df.y, p, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(df.y, p)),
            "ci": boot_ci(df, p, labels),
            "per_class": {NAMES[l]: float(v) for l, v in zip(labels, per)},
        }
        print(f"{name:38s} macroF1={results[name]['macro_f1']:.3f} CI={results[name]['ci'][0]:.3f}-{results[name]['ci'][1]:.3f} acc={results[name]['accuracy']:.3f}")
    pm = np.array([maj] * len(df), dtype=object)
    results["Majority class (always guess most common)"] = {
        "macro_f1": float(f1_score(df.y, pm, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(df.y, pm)), "ci": None, "per_class": {}}

    best = max((k for k in preds), key=lambda k: results[k]["macro_f1"])
    cm = confusion_matrix(df.y, preds[best], labels=labels)
    support = {NAMES[l]: int((df.y == l).sum()) for l in labels}
    json.dump({"n": len(df), "n_incidents": int(df["Incident Number"].nunique()), "labels": [NAMES[l] for l in labels],
               "support": support, "results": results, "best": best, "confusion": cm.tolist()},
              open(OUT / "eval.json", "w"), indent=2)
    o = df[["Incident Number", "Cause number", "src_text_stripped", "y"]].copy()
    for k, p in preds.items():
        o[k] = p
    o.to_csv(OUT / "oof.csv", index=False)

    # final models on all rows (for the app)
    v = tfidf().fit(text)
    m_t = lr(1.0).fit(v.transform(text), df.y)          # for word-level explanations
    m_b = lr(2.0).fit(hstack([v.transform(text), csr_matrix(E)]).tocsr(), df.y)
    m_e = lr(4.0).fit(E, df.y)                          # the winning model
    pickle.dump({"vec": v, "lr_tfidf": m_t, "lr_hybrid": m_b, "lr_emb": m_e, "names": NAMES,
                 "train_text": text, "train_y": df.y.tolist(), "train_E": E},
                open(OUT / "models.pkl", "wb"))
    print("best:", best)


if __name__ == "__main__":
    main()
