"""Stretch: LambdaMART (LightGBM LGBMRanker) re-ranking. Trained on the dev
split's hybrid top-100 candidates, evaluated on test.

Features per (query, candidate) pair:
  - bm25_score, bm25f_score      (lexical match strength)
  - cosine_sim                   (semantic match strength)
  - cpc_overlap                  (Jaccard over CPC codes -- same-subfield signal)
  - filing_date_proximity        (1 / (1 + |days between filings|))
  - cited_in_subset_count        (candidate's in-subset citation indegree --
                                   a citation-popularity prior independent of
                                   this specific query)
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import aggregate, evaluate_run
from src.eval.splits import load_split
from src.eval.trec_format import load_qrels, load_run, write_run
from src.index.bm25_index import load_index as load_bm25_idx
from src.index.bm25_index import tokenize
from src.index.bm25f_index import DEFAULT_FIELD_WEIGHTS
from src.index.bm25f_index import load_index as load_bm25f_idx
from src.index.vector_index import load_all as load_vector_all

FEATURE_NAMES = ["bm25_score", "bm25f_score", "cosine_sim", "cpc_overlap",
                  "filing_date_proximity", "cited_in_subset_count"]


class FeatureBuilder:
    def __init__(self, subset_path: str, citations_path: str,
                 bm25_index_path: str, bm25f_index_path: str, vector_index_dir: str):
        subset = pd.read_parquet(subset_path)
        subset["patent_id"] = subset["patent_id"].astype(str)
        self.subset = subset.set_index("patent_id")

        self.bm25, self.bm25_doc_ids = load_bm25_idx(bm25_index_path)
        self.bm25_doc_pos = {d: i for i, d in enumerate(self.bm25_doc_ids)}

        self.bm25f_fields, self.bm25f_doc_ids = load_bm25f_idx(bm25f_index_path)
        self.bm25f_doc_pos = {d: i for i, d in enumerate(self.bm25f_doc_ids)}

        flat_index, _ivf, self.vec_doc_ids, _model_name = load_vector_all(vector_index_dir)
        self.embeddings = np.load(os.path.join(vector_index_dir, "embeddings.npy"))
        self.vec_doc_pos = {d: i for i, d in enumerate(self.vec_doc_ids)}

        cited_counts = {}
        citation_edges = set()
        if os.path.exists(citations_path):
            citations = pd.read_parquet(citations_path)
            citations["citing_patent_number"] = citations["citing_patent_number"].astype(str)
            citations["cited_patent_number"] = citations["cited_patent_number"].astype(str)
            cited_counts = citations["cited_patent_number"].value_counts().to_dict()
            citation_edges = set(zip(citations["citing_patent_number"], citations["cited_patent_number"]))
        self.cited_counts = cited_counts
        self.citation_edges = citation_edges

    def query_vector(self, query_text: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        from src.index.vector_index import BGE_QUERY_PREFIX, EMBEDDING_MODEL
        if not hasattr(self, "_model"):
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model.encode([BGE_QUERY_PREFIX + query_text], normalize_embeddings=True,
                                   convert_to_numpy=True)[0].astype("float32")

    def query_score_arrays(self, query_tokens: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """BM25/BM25F get_scores() computes a score for the WHOLE corpus in one
        vectorized pass -- call this ONCE per query, not once per candidate
        doc (that would be O(top_n) redundant full-corpus scoring passes)."""
        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens))
        bm25f_scores = {f: np.asarray(bm25.get_scores(query_tokens)) for f, bm25 in self.bm25f_fields.items()}
        return bm25_scores, bm25f_scores

    def features(self, query_id: str, doc_id: str, bm25_scores: np.ndarray,
                 bm25f_scores: dict[str, np.ndarray], query_emb: np.ndarray) -> list[float]:
        bm25_score = float(bm25_scores[self.bm25_doc_pos[doc_id]]) if doc_id in self.bm25_doc_pos else 0.0

        bm25f_score = 0.0
        if doc_id in self.bm25f_doc_pos:
            pos = self.bm25f_doc_pos[doc_id]
            bm25f_score = sum(
                DEFAULT_FIELD_WEIGHTS.get(f, 1.0) * bm25f_scores[f][pos]
                for f in self.bm25f_fields
            )

        cosine_sim = 0.0
        if doc_id in self.vec_doc_pos:
            cosine_sim = float(np.dot(query_emb, self.embeddings[self.vec_doc_pos[doc_id]]))

        cpc_overlap = 0.0
        filing_date_proximity = 0.0
        if query_id in self.subset.index and doc_id in self.subset.index:
            q_row, d_row = self.subset.loc[query_id], self.subset.loc[doc_id]
            # cpc_codes round-trips through parquet as a numpy array, not a
            # plain list -- `arr or []` raises on multi-element arrays since
            # numpy arrays don't support simple truthiness. set() over it
            # (or an empty list) is unambiguous, so skip the `or` entirely.
            q_cpc, d_cpc = set(q_row["cpc_codes"]), set(d_row["cpc_codes"])
            if q_cpc or d_cpc:
                cpc_overlap = len(q_cpc & d_cpc) / max(1, len(q_cpc | d_cpc))
            if pd.notna(q_row["filing_date"]) and pd.notna(d_row["filing_date"]):
                days = abs((q_row["filing_date"] - d_row["filing_date"]).days)
                filing_date_proximity = 1.0 / (1.0 + days)

        # Leave-one-out: exclude the current query's own citation edge from
        # the count, or this feature directly leaks the label being
        # predicted (a true-positive doc's count would include the citation
        # FROM this very query, and in a sparse graph that one edge can
        # dominate a doc's total count).
        cited_in_subset_count = float(self.cited_counts.get(doc_id, 0))
        if (query_id, doc_id) in self.citation_edges:
            cited_in_subset_count -= 1.0

        return [bm25_score, bm25f_score, cosine_sim, cpc_overlap, filing_date_proximity, cited_in_subset_count]


def build_training_frame(feature_builder: FeatureBuilder, topics: dict[str, str],
                          hybrid_run: dict, qrels: dict, query_ids: list[str], top_n: int = 100) -> pd.DataFrame:
    rows = []
    for qid in query_ids:
        if qid not in topics:
            continue
        query_text = topics[qid]
        query_tokens = tokenize(query_text)
        query_emb = feature_builder.query_vector(query_text)
        bm25_scores, bm25f_scores = feature_builder.query_score_arrays(query_tokens)
        rel_map = qrels.get(qid, {})
        for doc_id, _score in hybrid_run.get(qid, [])[:top_n]:
            feats = feature_builder.features(qid, doc_id, bm25_scores, bm25f_scores, query_emb)
            rows.append({"query_id": qid, "doc_id": doc_id, "label": rel_map.get(doc_id, 0), **dict(zip(FEATURE_NAMES, feats))})
    return pd.DataFrame(rows)


def train_ltr(train_df: pd.DataFrame) -> lgb.LGBMRanker:
    train_df = train_df.sort_values("query_id")
    group_sizes = train_df.groupby("query_id").size().values
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=200,
                            learning_rate=0.05, num_leaves=15, verbose=-1)
    model.fit(train_df[FEATURE_NAMES], train_df["label"], group=group_sizes)
    return model


def apply_ltr(model: lgb.LGBMRanker, test_df: pd.DataFrame) -> dict:
    test_df = test_df.copy()
    test_df["ltr_score"] = model.predict(test_df[FEATURE_NAMES])
    run = {}
    for qid, group in test_df.groupby("query_id"):
        ranked = group.sort_values("ltr_score", ascending=False)
        run[qid] = list(zip(ranked["doc_id"], ranked["ltr_score"]))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--citations", default="data/qrels/raw_citations.parquet")
    ap.add_argument("--qrels", default="data/qrels/qrels.txt")
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--split-dir", default="data/qrels")
    ap.add_argument("--hybrid-run", default="results/runs/hybrid_rrf.run")
    ap.add_argument("--bm25-index", default="data/processed/bm25_index.pkl")
    ap.add_argument("--bm25f-index", default="data/processed/bm25f_index.pkl")
    ap.add_argument("--vector-index-dir", default="data/processed/vector_index")
    ap.add_argument("--out", default="results/runs/reranked_ltr.run")
    ap.add_argument("--feature-importance-out", default="results/ltr_feature_importance.png")
    args = ap.parse_args()

    from src.eval.trec_format import load_topics

    topics = load_topics(args.topics)
    qrels = load_qrels(args.qrels)
    hybrid_run = load_run(args.hybrid_run)
    dev_ids, test_ids = load_split(args.split_dir)

    fb = FeatureBuilder(args.subset, args.citations, args.bm25_index, args.bm25f_index, args.vector_index_dir)

    print("Building dev (train) features ...")
    train_df = build_training_frame(fb, topics, hybrid_run, qrels, dev_ids)
    print("Building test features ...")
    test_df = build_training_frame(fb, topics, hybrid_run, qrels, test_ids)

    model = train_ltr(train_df)
    test_run = apply_ltr(model, test_df)
    write_run(test_run, args.out, tag="ltr")
    print(f"Wrote {args.out}")

    avg = aggregate(evaluate_run(qrels, test_run))
    print(f"LTR test NDCG@10: {avg['ndcg_cut_10']:.4f}")

    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.barh(FEATURE_NAMES, importances)
    plt.xlabel("Importance")
    plt.title("LTR Feature Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.feature_importance_out), exist_ok=True)
    plt.savefig(args.feature_importance_out)
    print(f"Wrote {args.feature_importance_out}")


if __name__ == "__main__":
    main()
