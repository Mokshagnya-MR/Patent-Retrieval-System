"""Embedding + FAISS semantic search."""
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.vector_index import BGE_QUERY_PREFIX, EMBEDDING_MODEL, load_all

_state: dict = {}


def _get_state(index_dir: str, use_ivf: bool = False):
    if index_dir not in _state:
        flat_index, ivf_index, doc_ids, model_name = load_all(index_dir)
        model = SentenceTransformer(model_name)
        _state[index_dir] = {"flat": flat_index, "ivf": ivf_index, "doc_ids": doc_ids, "model": model}
    s = _state[index_dir]
    return (s["ivf"] if use_ivf else s["flat"]), s["doc_ids"], s["model"]


def search_semantic(query: str, k: int = 10, index_dir: str = "data/processed/vector_index",
                     use_ivf: bool = False) -> list[tuple[str, float]]:
    index, doc_ids, model = _get_state(index_dir, use_ivf)
    q_emb = model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idx = index.search(q_emb, k)
    return [(doc_ids[i], float(scores[0][rank])) for rank, i in enumerate(idx[0]) if i != -1]


def main():
    from src.eval.trec_format import load_topics, write_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--index-dir", default="data/processed/vector_index")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default="results/runs/semantic.run")
    ap.add_argument("--use-ivf", action="store_true")
    args = ap.parse_args()

    topics = load_topics(args.topics)
    run = {qid: search_semantic(text, args.k, args.index_dir, args.use_ivf) for qid, text in topics.items()}
    write_run(run, args.out, tag="semantic")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
