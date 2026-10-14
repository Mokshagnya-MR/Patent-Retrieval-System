"""K-fold cross-validated hybrid fusion over ALL topics.

The fixed 70/30 split leaves only 62 test queries, which is too few to
detect NDCG@10 differences of ~0.02-0.03 with a paired test (BM25 vs hybrid
is p~0.2 on the test split). Cross-validation keeps the no-tuning-on-test
guarantee -- alpha for each fold is chosen only on the other folds -- while
every one of the 206 queries is scored exactly once as a held-out query.

Writes results/cv_hybrid.csv (per-method means and paired Wilcoxon tests,
Holm-adjusted) and results/runs/hybrid_cv.run.

Usage: python -m src.eval.cross_validation
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

from src.eval.metrics import aggregate, evaluate_run
from src.eval.significance import holm_adjust, paired_test_runs
from src.eval.temporal_filter import filter_run
from src.eval.trec_format import load_qrels, load_run, write_run
from src.retrieve.hybrid import weighted_fusion

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def make_folds(qids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    qids = sorted(qids)
    random.Random(seed).shuffle(qids)
    return [qids[i::n_folds] for i in range(n_folds)]


def cross_validated_hybrid(bm25_run: dict, semantic_run: dict, qrels: dict, n_folds: int = 5,
                           seed: int = 42, depth: int = 100) -> tuple[dict, list[float]]:
    def fuse(qids, alpha):
        return {q: weighted_fusion(bm25_run.get(q, [])[:depth], semantic_run.get(q, [])[:depth], alpha)[:depth]
                for q in qids}

    cv_run, chosen = {}, []
    folds = make_folds(list(qrels), n_folds, seed)
    for i, held_out in enumerate(folds):
        train = [q for j, fold in enumerate(folds) if j != i for q in fold]
        best = max(ALPHAS, key=lambda a: (aggregate(evaluate_run(qrels, fuse(train, a)))["ndcg_cut_10"], -abs(a - 0.5)))
        chosen.append(float(best))
        cv_run.update(fuse(held_out, best))
    return cv_run, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-run", default="results/runs/bm25.run")
    ap.add_argument("--semantic-run", default="results/runs/semantic.run")
    ap.add_argument("--qrels", default="data/qrels/qrels.txt")
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cv_hybrid.csv")
    ap.add_argument("--run-out", default="results/runs/hybrid_cv.run")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    bm25_run, semantic_run = load_run(args.bm25_run), load_run(args.semantic_run)
    cv_run, chosen = cross_validated_hybrid(bm25_run, semantic_run, qrels, args.folds, args.seed)
    write_run(cv_run, args.run_out, tag="hybrid_cv")
    print(f"Alpha chosen per fold: {chosen}")

    subset = pd.read_parquet(args.subset, columns=["patent_id", "filing_date"])
    doc_dates = dict(zip(subset["patent_id"].astype(str), subset["filing_date"]))
    runs = {"BM25": bm25_run, "Semantic": semantic_run, "Hybrid (CV alpha)": cv_run,
            "Hybrid (CV alpha) + prior-art filter": filter_run(cv_run, doc_dates)}

    rows = []
    for name, run in runs.items():
        avg = aggregate(evaluate_run(qrels, run))
        rows.append({"row_type": "method", "name": name, "P@10": avg["P_10"], "R@10": avg["recall_10"],
                     "MRR": avg["recip_rank"], "NDCG@10": avg["ndcg_cut_10"]})
    comparisons = [("BM25", "Hybrid (CV alpha)"), ("Semantic", "Hybrid (CV alpha)"),
                   ("Hybrid (CV alpha)", "Hybrid (CV alpha) + prior-art filter")]
    tests = [paired_test_runs(runs[a], runs[b], qrels, "ndcg_cut_10") for a, b in comparisons]
    for (a, b), t, p_holm in zip(comparisons, tests, holm_adjust([t["p_value"] for t in tests])):
        rows.append({"row_type": "test", "name": f"{a} vs {b}", "n": t["n"], "mean_a": t["mean_a"],
                     "mean_b": t["mean_b"], "p_value": t["p_value"], "p_value_holm": p_holm})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(4))
    print(f"Wrote {args.out} and {args.run_out}")


if __name__ == "__main__":
    main()
