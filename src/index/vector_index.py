"""Embed the corpus and build FAISS indices: exact (IndexFlatIP) for
correctness, and an approximate IndexIVFFlat for the recall-vs-latency
comparison in Phase 4's DoD (see src/eval/ann_tradeoff.py).

Model default: BAAI/bge-small-en-v1.5 (384-dim, fast on CPU). Swap-in path
for a larger model: change EMBEDDING_MODEL and re-run -- everything else
(index building, search) is agnostic to dimensionality.
"""
import argparse
import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.preprocess.clean_text import clean_field

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on retrieval queries (not on indexed passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_corpus_text(subset: pd.DataFrame) -> list[str]:
    title = subset["title"].fillna("").map(clean_field)
    abstract = subset["abstract"].fillna("").map(clean_field)
    return (title + ". " + abstract).tolist()


def embed_corpus(texts: list[str], model_name: str = EMBEDDING_MODEL, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int = 100, nprobe: int = 10) -> faiss.Index:
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, max(1, len(embeddings) // 39)), faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = nprobe
    return index


def save_all(embeddings: np.ndarray, flat_index: faiss.Index, ivf_index: faiss.Index,
             doc_ids: list[str], out_dir: str, model_name: str = EMBEDDING_MODEL):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    faiss.write_index(flat_index, os.path.join(out_dir, "flat.index"))
    faiss.write_index(ivf_index, os.path.join(out_dir, "ivf.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "model_name": model_name}, f)


def load_all(out_dir: str):
    flat_index = faiss.read_index(os.path.join(out_dir, "flat.index"))
    ivf_index = faiss.read_index(os.path.join(out_dir, "ivf.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return flat_index, ivf_index, meta["doc_ids"], meta["model_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--out-dir", default="data/processed/vector_index")
    ap.add_argument("--model", default=EMBEDDING_MODEL)
    args = ap.parse_args()

    subset = pd.read_parquet(args.subset)
    texts = build_corpus_text(subset)
    doc_ids = subset["patent_id"].astype(str).tolist()

    print(f"Embedding {len(texts)} docs with {args.model} ...")
    embeddings = embed_corpus(texts, args.model)

    flat_index = build_flat_index(embeddings)
    ivf_index = build_ivf_index(embeddings)

    save_all(embeddings, flat_index, ivf_index, doc_ids, args.out_dir, args.model)
    print(f"Saved vector indices to {args.out_dir}")


if __name__ == "__main__":
    main()
