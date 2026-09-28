"""Phase 6 ablation: patent_full vs patent_reduced query construction,
scored with BM25 and hybrid retrieval, on the patent-as-query topics (every
topic already IS a patent -- our qrels are citation-derived, so the query
is naturally "this patent" rather than free text)."""
import argparse

import pandas as pd

from src.eval.metrics import aggregate, evaluate_run
from src.eval.trec_format import exclude_self_match, load_qrels, load_run, write_run
from src.retrieve.baseline import search_bm25
from src.retrieve.hybrid import weighted_fusion
from src.retrieve.query_reduction import TfidfQueryReducer, build_full_query, build_query, reduce_simple
from src.retrieve.semantic import search_semantic


def run_ablation(subset_path: str, qrels_path: str, semantic_index_dir: str,
                  bm25_index_path: str, k: int = 100, reduced_variant: str = "tfidf") -> pd.DataFrame:
    subset = pd.read_parquet(subset_path)
    subset["patent_id"] = subset["patent_id"].astype(str)
    subset_indexed = subset.set_index("patent_id")
    qrels = load_qrels(qrels_path)
    query_ids = [qid for qid in qrels if qid in subset_indexed.index]

    tfidf_reducer = None
    if reduced_variant == "tfidf":
        corpus_texts = [build_full_query(row) for _, row in subset.iterrows()]
        tfidf_reducer = TfidfQueryReducer(corpus_texts, top_n=40)

    rows = []
    for mode in ("patent_full", "patent_reduced"):
        bm25_run, hybrid_run = {}, {}
        for qid in query_ids:
            row = subset_indexed.loc[qid].to_dict()
            query_text = build_query(row, mode, tfidf_reducer) if mode == "patent_reduced" and reduced_variant == "tfidf" \
                else (build_full_query(row) if mode == "patent_full" else reduce_simple(row))

            # Patent-as-query: over-fetch by 1 and drop the trivial self-match
            # (the query text IS this document's own title+abstract/claims).
            bm25_results = exclude_self_match(search_bm25(query_text, k + 1, bm25_index_path), qid, k)
            sem_results = exclude_self_match(search_semantic(query_text, k + 1, semantic_index_dir), qid, k)
            bm25_run[qid] = bm25_results
            hybrid_run[qid] = weighted_fusion(bm25_results, sem_results, alpha=0.5)[:k]

        for method_name, run in [("bm25", bm25_run), ("hybrid", hybrid_run)]:
            avg = aggregate(evaluate_run(qrels, run))
            rows.append({"query_mode": mode, "method": method_name, "ndcg_10": avg["ndcg_cut_10"],
                         "p_10": avg["P_10"], "recall_10": avg["recall_10"]})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--qrels", default="data/qrels/qrels.txt")
    ap.add_argument("--semantic-index-dir", default="data/processed/vector_index")
    ap.add_argument("--bm25-index", default="data/processed/bm25_index.pkl")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--reduced-variant", choices=["simple", "tfidf"], default="tfidf")
    ap.add_argument("--out", default="results/query_reduction_ablation.csv")
    args = ap.parse_args()

    df = run_ablation(args.subset, args.qrels, args.semantic_index_dir, args.bm25_index, args.k, args.reduced_variant)
    df.to_csv(args.out, index=False)
    print(df)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
