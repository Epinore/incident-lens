"""Compare the model with our team's hand labels (labeling/gold_to_label.csv).

Rows that were in the training set use their out-of-fold prediction (no leakage);
other rows use the final model. Writes artifacts/gold.json.
"""
from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CODE2EL = {"HU": "3", "EQ": "15", "MS": "8", "SU": "17", "CO": "9", "WE": "6"}


def main():
    gd = pd.read_csv(HERE / "labeling" / "gold_to_label.csv", encoding="utf-8-sig")
    gd["lab"] = gd.my_label.astype(str).str.upper().str.strip()
    gd = gd[gd.lab.isin(CODE2EL)].copy()
    out = {"n": int(len(gd))}
    if len(gd):
        M = pickle.load(open(HERE / "artifacts" / "models.pkl", "rb"))
        ev = json.load(open(HERE / "artifacts" / "eval.json"))
        oof = pd.read_csv(HERE / "artifacts" / "oof.csv", dtype={"y": str})
        oof["rid"] = oof["Incident Number"] + "-" + oof["Cause number"].astype(str)
        oofmap = dict(zip(oof.rid, oof[ev["best"]].astype(str)))
        from fastembed import TextEmbedding
        m = TextEmbedding("BAAI/bge-small-en-v1.5")
        E = np.array(list(m.embed(gd.text.fillna("").astype(str).tolist())), dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        fresh = M["lr_emb"].predict(E).astype(str)
        gd["model"] = [oofmap.get(r, f) for r, f in zip(gd.report_id, fresh)]
        gd["el"] = gd.lab.map(CODE2EL)
        out["agree"] = int((gd.el == gd.model).sum())
        out["miss_examples"] = [{"ours": M["names"][a], "model": M["names"][b]}
                                for a, b in zip(gd.el, gd.model) if a != b][:5]
    json.dump(out, open(HERE / "artifacts" / "gold.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
