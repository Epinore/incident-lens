"""Compare the cause model with our team's hand labels.

1. Import the labels exported from the team Google Sheet into labeling/gold_to_label.csv by report_id
   (local text is kept; my_label and notes come from the sheet).
2. For every labeled row, predict with a model that never saw that INCIDENT:
   - incident is in the 374-row training set -> refit the embedding model on all folds except the
     incident's syn_fold (folds are grouped by incident, so every row of that incident is held out);
   - incident not in the training set -> the final model (it never saw the incident).
3. Report valid n, "?" count, label distribution, agreement (bootstrap CI over incidents), per class,
   and every disagreement. Writes artifacts/gold.json and labeling/gold_disagreements.csv.

These labels are used ONLY for this check. The model is not tuned on them.
"""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
TASK = Path(r"D:\Projects\psm-incident-ml\data\processed\e19\cause_task.csv")
CODE2EL = {"HU": "3", "EQ": "15", "MS": "8", "SU": "17", "CO": "9", "WE": "6"}
SEED = 0


def import_sheet():
    gold_p = HERE / "labeling" / "gold_to_label.csv"
    gd = pd.read_csv(gold_p, encoding="utf-8-sig", dtype=str).fillna("")
    sheets = sorted((HERE / "labeling").glob("sheet_export_*.json"))
    sh = json.load(open(sheets[-1], encoding="utf-8"))
    s = pd.DataFrame(sh["rows"], columns=sh["columns"])
    missing = set(s.report_id) - set(gd.report_id)
    assert not missing, f"sheet ids not in local CSV: {missing}"
    m = gd.merge(s, on="report_id", how="left", suffixes=("_old", ""))
    bad_len = m[(m.text_len.notna()) & (m.text.str.len() != m.text_len)]
    gd["my_label"] = m["my_label"].fillna("").values
    gd["notes"] = m["notes"].fillna("").values
    gd.to_csv(gold_p, index=False, encoding="utf-8-sig")
    return gd, sheets[-1].name, bad_len.report_id.tolist()


def embed(texts):
    from fastembed import TextEmbedding
    e = np.array(list(TextEmbedding("BAAI/bge-small-en-v1.5").embed(texts, batch_size=32)), dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    gd, src, bad_len = import_sheet()
    M = pickle.load(open(HERE / "artifacts" / "models.pkl", "rb"))
    names = M["names"]
    gd["lab"] = gd.my_label.str.upper().str.strip()
    n_q = int((gd.lab == "?").sum())
    n_blank = int((gd.lab == "").sum())
    v = gd[gd.lab.isin(CODE2EL)].copy()
    v["el"] = v.lab.map(CODE2EL)
    v["incident"] = v.report_id.str.rsplit("-", n=1).str[0]

    tr = pd.read_csv(TASK, dtype={"xw_element": str})
    tr = tr[tr.syn_regime == "modern_six"].reset_index(drop=True)
    tr["rid"] = tr["Incident Number"] + "-" + tr["Cause number"].astype(str)
    E_tr = M["train_E"]
    y_tr = np.array(M["train_y"])
    inc_fold = dict(zip(tr["Incident Number"], tr.syn_fold))
    inv_label = dict(zip(tr.rid, tr.xw_element))

    E = embed(v.text.tolist())
    preds, how = [], []
    fold_models = {}
    for i, inc in enumerate(v.incident):
        if inc in inc_fold:
            f = inc_fold[inc]
            if f not in fold_models:
                keep = (tr.syn_fold != f).values
                fold_models[f] = LogisticRegression(class_weight="balanced", max_iter=3000, C=4.0,
                                                    random_state=SEED).fit(E_tr[keep], y_tr[keep])
            preds.append(fold_models[f].predict(E[i:i + 1])[0]); how.append(f"held-out fold {f}")
        else:
            preds.append(M["lr_emb"].predict(E[i:i + 1])[0]); how.append("incident not in training")
    v["model"] = [str(p) for p in preds]
    v["how"] = how
    v["investigator"] = v.report_id.map(inv_label)
    # sanity: no prediction came from a model that trained on that incident
    for inc, h in zip(v.incident, v.how):
        if h.startswith("held-out"):
            f = int(h.split()[-1])
            assert not ((tr["Incident Number"] == inc) & (tr.syn_fold != f)).any()

    v["agree"] = v.el == v.model
    rng = np.random.default_rng(SEED)
    incs = v.incident.unique()
    by = {k: g.agree.values for k, g in v.groupby("incident")}
    boots = [np.concatenate([by[k] for k in rng.choice(incs, len(incs))]).mean() for _ in range(2000)]
    per = {names[c]: {"n": int((v.el == c).sum()), "agree": int(((v.el == c) & v.agree).sum())}
           for c in CODE2EL.values() if (v.el == c).any()}
    labs = list(CODE2EL.values())
    conf = pd.crosstab(v.el.map(names), v.model.map(names)).reindex(index=[names[c] for c in labs],
                                                                  columns=[names[c] for c in labs], fill_value=0)
    inv = v[v.investigator.notna()]
    out = {
        "source": src, "rows_in_sheet": int(len(gd)), "n": int(len(v)), "n_question": n_q, "n_blank": n_blank,
        "n_incidents": int(v.incident.nunique()),
        "agree": int(v.agree.sum()), "agree_rate": round(float(v.agree.mean()), 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "prediction_source": v.how.str.replace(r" \d+$", "", regex=True).value_counts().to_dict(),
        "distribution": {names[k]: int(c) for k, c in v.el.value_counts().items()},
        "per_class": per,
        "confusion_rows_team_cols_model": conf.to_dict(orient="index"),
        "team_vs_investigator": {"n": int(len(inv)), "agree": int((inv.el == inv.investigator).sum())},
        "text_length_mismatch_vs_sheet": bad_len,
        "notes_nonempty": int((gd.notes.str.strip() != "").sum()),
        "miss_examples": [{"ours": names[a], "model": names[b]} for a, b in zip(v.el, v.model) if a != b][:5],
    }
    json.dump(out, open(HERE / "artifacts" / "gold.json", "w"), indent=2)
    d = v[~v.agree].assign(team=lambda x: x.el.map(names), model_pred=lambda x: x.model.map(names),
                           investigator_label=lambda x: x.investigator.map(names))
    d[["report_id", "team", "model_pred", "investigator_label", "how", "notes", "text"]].to_csv(
        HERE / "labeling" / "gold_disagreements.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(out, indent=2))
    pd.set_option("display.width", 250, "display.max_colwidth", 110)
    print(d[["report_id", "team", "model_pred", "investigator_label", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
