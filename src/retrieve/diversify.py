"""Result diversification: patent-family dedup (title/abstract near-duplicate
clustering) applied as a post-processing step on a ranked run. Diversification
trades relevance for coverage by design, so we don't re-score it against
qrels -- it's evaluated qualitatively (before/after examples)."""
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.85


def dedup_family(ranked: list[tuple[str, float]], doc_texts: dict[str, str],
                  threshold: float = SIMILARITY_THRESHOLD) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    """Collapse near-duplicate (title+abstract) results, keeping the
    highest-ranked representative of each cluster. Returns the diversified
    ranking plus {representative_doc_id: [collapsed_doc_ids]}."""
    doc_ids = [d for d, _ in ranked]
    texts = [doc_texts.get(d, "") for d in doc_ids]
    if len(texts) < 2:
        return ranked, {}

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf)

    kept: list[tuple[str, float]] = []
    kept_idx: list[int] = []
    families: dict[str, list[str]] = {}
    for i, (doc_id, score) in enumerate(ranked):
        collapsed_into = None
        for j in kept_idx:
            if sim[i][j] >= threshold:
                collapsed_into = doc_ids[j]
                break
        if collapsed_into is None:
            kept.append((doc_id, score))
            kept_idx.append(i)
            families[doc_id] = []
        else:
            families[collapsed_into].append(doc_id)
    return kept, families


def mmr(ranked: list[tuple[str, float]], doc_embeddings: dict, lambda_param: float = 0.7,
        top_k: int = 10) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance re-ranking of the top candidates, trading
    relevance (lambda_param) against novelty vs. already-selected results."""
    import numpy as np

    candidates = [(d, s) for d, s in ranked if d in doc_embeddings]
    if not candidates:
        return ranked[:top_k]

    selected: list[tuple[str, float]] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            def mmr_score(item):
                doc_id, rel_score = item
                max_sim = max(
                    float(np.dot(doc_embeddings[doc_id], doc_embeddings[s_id]))
                    for s_id, _ in selected
                )
                return lambda_param * rel_score - (1 - lambda_param) * max_sim
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def main():
    import pandas as pd

    from src.eval.trec_format import load_run, write_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/reranked.run")
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--out", default="results/runs/diversified.run")
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = ap.parse_args()

    run = load_run(args.run)
    subset = pd.read_parquet(args.subset)
    subset["patent_id"] = subset["patent_id"].astype(str)
    doc_texts = {row["patent_id"]: f"{row['title']}. {row['abstract']}" for _, row in subset.iterrows()}

    diversified_run = {}
    for qid, ranked in run.items():
        diversified, families = dedup_family(ranked, doc_texts, args.threshold)
        diversified_run[qid] = diversified
        collapsed = {k: v for k, v in families.items() if v}
        if collapsed:
            print(f"Query {qid}: collapsed {sum(len(v) for v in collapsed.values())} near-duplicates -> {collapsed}")

    write_run(diversified_run, args.out, tag="diversified")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
