"""Field-weighted BM25 (BM25F-style): a separate BM25 index per field
(title, abstract, claims), combined at query time with per-field weights.
Claims are weighted highest by default -- they define the patent's legal
scope and are the field most predictive of true topical relevance."""
import argparse
import os
import pickle

import pandas as pd
from rank_bm25 import BM25Okapi

from src.index.bm25_index import tokenize
from src.preprocess.clean_text import clean_field

FIELDS = ("title", "abstract", "claims")
DEFAULT_FIELD_WEIGHTS = {"title": 1.0, "abstract": 1.0, "claims": 2.0}


def build_bm25f_index(subset: pd.DataFrame, fields=FIELDS):
    field_indices = {}
    for field in fields:
        strip_claims = field == "claims"
        texts = subset[field].fillna("").map(lambda t: clean_field(t, strip_claims=strip_claims))
        tokenized = [tokenize(t) for t in texts]
        field_indices[field] = BM25Okapi(tokenized)
    doc_ids = subset["patent_id"].astype(str).tolist()
    return field_indices, doc_ids


def save_index(field_indices, doc_ids, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"field_indices": field_indices, "doc_ids": doc_ids}, f)


def load_index(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["field_indices"], data["doc_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--output", default="data/processed/bm25f_index.pkl")
    args = ap.parse_args()

    subset = pd.read_parquet(args.subset)
    field_indices, doc_ids = build_bm25f_index(subset)
    save_index(field_indices, doc_ids, args.output)
    print(f"Built BM25F index over {len(doc_ids)} docs, fields={list(field_indices)} -> {args.output}")


if __name__ == "__main__":
    main()
