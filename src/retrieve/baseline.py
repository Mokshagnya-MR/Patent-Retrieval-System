"""TF-IDF/BM25 baseline retrieval: search_bm25 and search_bm25f."""
import argparse

import numpy as np

from src.eval.trec_format import load_topics, write_run
from src.index.bm25_index import load_index as load_bm25, tokenize
from src.index.bm25f_index import DEFAULT_FIELD_WEIGHTS
from src.index.bm25f_index import load_index as load_bm25f

_bm25_cache: dict = {}
_bm25f_cache: dict = {}


def search_bm25(query: str, k: int = 10, index_path: str = "data/processed/bm25_index.pkl"):
    if index_path not in _bm25_cache:
        _bm25_cache[index_path] = load_bm25(index_path)
    bm25, doc_ids = _bm25_cache[index_path]

    scores = np.asarray(bm25.get_scores(tokenize(query)))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(doc_ids[i], float(scores[i])) for i in top_idx]


def search_bm25f(query: str, k: int = 10, weights: dict | None = None,
                  index_path: str = "data/processed/bm25f_index.pkl"):
    if index_path not in _bm25f_cache:
        _bm25f_cache[index_path] = load_bm25f(index_path)
    field_indices, doc_ids = _bm25f_cache[index_path]
    weights = weights or DEFAULT_FIELD_WEIGHTS

    q_tokens = tokenize(query)
    total = np.zeros(len(doc_ids))
    for field, bm25 in field_indices.items():
        total += weights.get(field, 1.0) * np.asarray(bm25.get_scores(q_tokens))

    top_idx = np.argsort(total)[::-1][:k]
    return [(doc_ids[i], float(total[i])) for i in top_idx]


def run_over_topics(topics: dict[str, str], search_fn, k: int = 100) -> dict[str, list[tuple[str, float]]]:
    run = {}
    for qid, text in topics.items():
        run[qid] = search_fn(text, k)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--bm25-index", default="data/processed/bm25_index.pkl")
    ap.add_argument("--bm25f-index", default="data/processed/bm25f_index.pkl")
    ap.add_argument("--bm25-run-out", default="results/runs/bm25.run")
    ap.add_argument("--bm25f-run-out", default="results/runs/bm25f.run")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    print(f"Loaded {len(topics)} topics")

    bm25_run = run_over_topics(topics, lambda q, k: search_bm25(q, k, args.bm25_index), args.k)
    write_run(bm25_run, args.bm25_run_out, tag="bm25")
    print(f"Wrote {args.bm25_run_out}")

    bm25f_run = run_over_topics(topics, lambda q, k: search_bm25f(q, k, index_path=args.bm25f_index), args.k)
    write_run(bm25f_run, args.bm25f_run_out, tag="bm25f")
    print(f"Wrote {args.bm25f_run_out}")


if __name__ == "__main__":
    main()
