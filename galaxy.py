"""Galaxy: 2-D map of all incidents by meaning + auto-discovered themes.

t-SNE on the embeddings for layout; KMeans on the embeddings for themes; each theme is
named by its most distinctive words (TF-IDF over the theme's narratives). Writes artifacts/galaxy.pkl.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ART = Path(__file__).parent / "artifacts"
K = 12
STOP = {"ip", "injured", "person", "incident", "occurred", "time", "approximately", "platform", "rig", "the", "was",
        "were", "employee", "operator", "contract", "contractor", "crew", "work", "working", "member", "hand", "left", "right", "mississippi", "canyon", "ocs", "inch", "bsee", "gulf", "mexico", "block", "green", "ewing", "bank", "garden", "banks", "viosca", "knoll", "eugene", "island", "ship", "shoal", "south", "west", "east", "main", "pass", "timbalier", "vermilion", "marsh", "high", "galveston", "shell", "operated", "exploration", "company", "llc", "inc", "energy", "offshore", "production", "operating"}


def main():
    inc = pd.read_pickle(ART / "incidents.pkl")
    E = np.load(ART / "emb.npy")
    xy = TSNE(n_components=2, perplexity=35, init="pca", random_state=0, metric="cosine").fit_transform(E)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(E)
    lab = km.labels_
    vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.4, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    X = vec.fit_transform(inc.what)
    vocab = np.array(vec.get_feature_names_out())
    names = {}
    for c in range(K):
        m = np.asarray(X[lab == c].mean(axis=0)).ravel() - np.asarray(X[lab != c].mean(axis=0)).ravel()
        words = [w for w in vocab[np.argsort(-m)] if not (set(w.split()) & STOP)][:3]
        names[c] = " · ".join(words)
    inc = inc.assign(x=xy[:, 0], y=xy[:, 1], theme=[names[c] for c in lab])
    # Plain-English names we gave each theme after reading its top words (keywords kept alongside)
    FRIENDLY = {"drill": "Drill pipe handling", "grating": "Falls & deck openings", "compressor": "Engines & compressors",
                "crane": "Crane & lifting", "injury": "Hand & finger injuries", "mud": "Drilling mud releases",
                "gas": "Gas & pressure releases", "oil": "Oil sheens & pipeline leaks", "vessel": "Vessel strikes",
                "casing": "Casing running", "smoke": "Small fires & smoke", "eni": "Short admin-style reports"}
    inc["keywords"] = inc.theme
    inc["theme"] = [FRIENDLY.get(t.split(" · ")[0], t) for t in inc.theme]
    inc[["id", "x", "y", "theme", "keywords"]].to_pickle(ART / "galaxy.pkl")
    print(pd.Series(lab).map(names).value_counts().to_string())


if __name__ == "__main__":
    main()
