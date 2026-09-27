"""Hybrid fusion of BM25 (or BM25F) and semantic retrieval: RRF (no tuning)
and weighted linear combination (alpha swept on a dev split, see
Phase 5 ablation script run_alpha_sweep.py)."""
import argparse

import numpy as np


def reciprocal_rank_fusion(*rankings: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: each a list of (doc_id, score) already sorted best-first.
    score(d) = sum over rankings of 1/(k + rank_i(d)), rank is 1-indexed."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _minmax_normalize(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    scores = np.array([s for _, s in pairs], dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return {doc_id: 0.0 for doc_id, _ in pairs}
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in pairs}


def weighted_fusion(bm25_results: list[tuple[str, float]], semantic_results: list[tuple[str, float]],
                     alpha: float = 0.5) -> list[tuple[str, float]]:
    """score(d) = alpha * norm(bm25) + (1 - alpha) * norm(cosine).
    alpha=1.0 -> pure BM25, alpha=0.0 -> pure semantic."""
    bm25_norm = _minmax_normalize(bm25_results)
    sem_norm = _minmax_normalize(semantic_results)
    all_docs = set(bm25_norm) | set(sem_norm)
    combined = {
        doc_id: alpha * bm25_norm.get(doc_id, 0.0) + (1 - alpha) * sem_norm.get(doc_id, 0.0)
        for doc_id in all_docs
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def main():
    """Build the RRF and dev-alpha-selected weighted-fusion run files, given
    already-built bm25 and semantic run files (run wider than k=10, e.g.
    k=100, so fusion has enough candidates from each side to combine)."""
    from src.eval.trec_format import load_run, load_topics, write_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-run", default="results/runs/bm25.run")
    ap.add_argument("--semantic-run", default="results/runs/semantic.run")
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rrf-out", default="results/runs/hybrid_rrf.run")
    ap.add_argument("--weighted-out", default="results/runs/hybrid.run")
    args = ap.parse_args()

    bm25_run = load_run(args.bm25_run)
    semantic_run = load_run(args.semantic_run)
    topics = load_topics(args.topics)

    rrf_run, weighted_run = {}, {}
    for qid in topics:
        bm25_results = bm25_run.get(qid, [])[: args.k]
        sem_results = semantic_run.get(qid, [])[: args.k]
        rrf_run[qid] = reciprocal_rank_fusion(bm25_results, sem_results)[: args.k]
        weighted_run[qid] = weighted_fusion(bm25_results, sem_results, args.alpha)[: args.k]

    write_run(rrf_run, args.rrf_out, tag="hybrid_rrf")
    write_run(weighted_run, args.weighted_out, tag=f"hybrid_alpha{args.alpha}")
    print(f"Wrote {args.rrf_out} and {args.weighted_out}")


if __name__ == "__main__":
    main()
