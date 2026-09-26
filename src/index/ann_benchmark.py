"""Benchmark exact (IndexFlatIP) vs approximate (IndexIVFFlat) search:
query latency and recall@10-vs-exact (using the exact index's top-10 as the
gold standard for the ANN index to be measured against, not the qrels --
this isolates the ANN-approximation cost from retrieval quality itself)."""
import argparse
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.index.vector_index import BGE_QUERY_PREFIX, load_all


def benchmark(index_dir: str, topics: dict[str, str], k: int = 10) -> pd.DataFrame:
    flat_index, ivf_index, doc_ids, model_name = load_all(index_dir)
    model = SentenceTransformer(model_name)

    rows = []
    for qid, text in topics.items():
        q_emb = model.encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

        t0 = time.perf_counter()
        _, flat_idx = flat_index.search(q_emb, k)
        flat_latency = time.perf_counter() - t0
        flat_ids = set(doc_ids[i] for i in flat_idx[0] if i != -1)

        t0 = time.perf_counter()
        _, ivf_idx = ivf_index.search(q_emb, k)
        ivf_latency = time.perf_counter() - t0
        ivf_ids = set(doc_ids[i] for i in ivf_idx[0] if i != -1)

        recall_vs_exact = len(flat_ids & ivf_ids) / max(1, len(flat_ids))
        rows.append({
            "query_id": qid,
            "flat_latency_ms": flat_latency * 1000,
            "ivf_latency_ms": ivf_latency * 1000,
            f"ivf_recall_at_{k}_vs_exact": recall_vs_exact,
        })
    return pd.DataFrame(rows)


def main():
    from src.eval.trec_format import load_topics

    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--index-dir", default="data/processed/vector_index")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="results/ann_tradeoff.csv")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    df = benchmark(args.index_dir, topics, args.k)
    df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}")
    print(df.drop(columns=["query_id"]).mean())


if __name__ == "__main__":
    main()
