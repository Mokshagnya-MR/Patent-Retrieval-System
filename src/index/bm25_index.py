"""Plain BM25 index over title+abstract+claims concatenated."""
import argparse
import os
import pickle
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocess.clean_text import clean_field

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    """Keep alphanumerics and internal hyphens intact so chemical/alphanumeric
    tokens (LiFePO4, Cu(NO3)2 -> 'cu', 'no3', '2') survive as meaningful
    units rather than being shredded by a stricter word tokenizer."""
    if not isinstance(text, str):
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_corpus_text(subset: pd.DataFrame) -> pd.Series:
    title = subset["title"].fillna("").map(lambda t: clean_field(t))
    abstract = subset["abstract"].fillna("").map(lambda t: clean_field(t))
    claims = subset["claims"].fillna("").map(lambda t: clean_field(t, strip_claims=True))
    return title + " " + abstract + " " + claims


def build_bm25_index(subset: pd.DataFrame):
    texts = build_corpus_text(subset)
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    doc_ids = subset["patent_id"].astype(str).tolist()
    return bm25, doc_ids


def save_index(bm25, doc_ids, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)


def load_index(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--output", default="data/processed/bm25_index.pkl")
    args = ap.parse_args()

    subset = pd.read_parquet(args.subset)
    bm25, doc_ids = build_bm25_index(subset)
    save_index(bm25, doc_ids, args.output)
    print(f"Built BM25 index over {len(doc_ids)} docs -> {args.output}")


if __name__ == "__main__":
    main()
