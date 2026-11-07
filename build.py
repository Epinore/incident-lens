"""Build step: load real_only tables, assemble one record per incident, embed with a local model.

Run:  uv run python build.py
Writes artifacts/ (incidents.pkl, emb.npy). Uses ONLY real_only/ (synthetic cells blanked).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA = Path(r"D:\Projects\psm-incident-ml\data\processed\e19")
OUT = Path(__file__).parent / "artifacts"
OUT.mkdir(exist_ok=True)

# crude person-name scrub for display: titles + capitalised name
NAME_RX = re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Capt|Captain)\.?\s+[A-Z][a-zA-Z'\-]+(\s+[A-Z][a-zA-Z'\-]+)?")


def scrub(t: str) -> str:
    return NAME_RX.sub("[name removed]", t or "")


def clean(t) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()


def main():
    inc = pd.read_csv(DATA / "real_only" / "incidents.csv")
    inc.columns = [c.strip() for c in inc.columns]
    cau = pd.read_csv(DATA / "real_only" / "causes.csv")
    cau.columns = [c.strip() for c in cau.columns]
    rec = pd.read_csv(DATA / "real_only" / "recommendations.csv")
    rec.columns = [c.strip() for c in rec.columns]

    causes = cau.groupby("Incident Number")["Cause Description"].apply(
        lambda s: [scrub(clean(x)) for x in s if clean(x)]).to_dict()
    recs = rec.groupby("Incident Number")["Recommendation Description"].apply(
        lambda s: [scrub(clean(x)) for x in s if clean(x)]).to_dict()

    rows = []
    for _, r in inc.iterrows():
        k = r["Incident Number"]
        what = clean(r.get("What happened?"))
        rows.append({
            "id": k,
            "date": clean(r.get("Date of Incident")),
            "year": int(str(r["Date of Incident"])[:4]) if isinstance(r.get("Date of Incident"), str) else None,
            "area": clean(r.get("Site")),
            "type": clean(r.get("Description")).title(),
            "event": " / ".join(x for x in [clean(r.get(f"Incident Type {c}")) for c in "ABCD"] if x),
            "what": scrub(what),
            "outcome": scrub(clean(r.get("What was the outcome?"))),
            "causes": causes.get(k, []),
            "recs": recs.get(k, []),
        })
    df = pd.DataFrame(rows)
    df = df[df.what.str.len() > 40].reset_index(drop=True)
    print("incidents with narrative:", len(df))

    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    texts = (df.type + ". " + df.what.str[:2000]).tolist()
    emb = np.array(list(model.embed(texts, batch_size=32)), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(OUT / "emb.npy", emb)
    df.to_pickle(OUT / "incidents.pkl")
    print("saved", emb.shape)


if __name__ == "__main__":
    main()
