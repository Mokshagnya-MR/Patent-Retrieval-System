"""Temporal prior-art filter evaluation: re-score every pipeline run after
dropping retrieved docs filed AFTER the query patent (they cannot be prior
art for it -- see src/retrieve/temporal.py), and test whether the change is
significant per method.

Also reports how much of the qrels themselves the filter makes unreachable
(citations to later-filed continuation / co-pending documents), i.e. the
recall ceiling the filter imposes, so the gain isn't overstated.

Usage: python -m src.eval.temporal_filter
"""
import argparse
import os

import pandas as pd

from src.eval.final_comparison import PIPELINE
from src.eval.metrics import aggregate, evaluate_run
from src.eval.significance import holm_adjust, paired_test_runs
from src.eval.trec_format import load_qrels, load_run
from src.retrieve.temporal import filter_prior_art


def filter_run(run: dict, doc_dates: dict) -> dict:
    return {qid: filter_prior_art(ranked, doc_dates.get(qid), doc_dates) for qid, ranked in run.items()}


def later_filed_at_k(run: dict, qrels: dict, doc_dates: dict, k: int = 10) -> float:
    counts = [
        sum(1 for d, _ in ranked[:k] if doc_dates.get(d) is not None and doc_dates[d] > doc_dates[qid])
        for qid, ranked in run.items() if qid in qrels and qid in doc_dates
    ]
    return sum(counts) / len(counts) if counts else 0.0


def qrels_ceiling(qrels: dict, doc_dates: dict) -> tuple[int, int]:
    """(qrels pairs made unreachable by the filter, total qrels pairs)."""
    pairs = [(q, d) for q, docs in qrels.items() for d in docs]
    lost = sum(1 for q, d in pairs if d in doc_dates and q in doc_dates and doc_dates[d] > doc_dates[q])
    return lost, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--qrels", default="data/qrels/qrels_test.txt")
    ap.add_argument("--out", default="results/temporal_filter.csv")
    args = ap.parse_args()

    subset = pd.read_parquet(args.subset, columns=["patent_id", "filing_date"])
    doc_dates = dict(zip(subset["patent_id"].astype(str), subset["filing_date"]))
    qrels = load_qrels(args.qrels)

    lost, total = qrels_ceiling(qrels, doc_dates)
    print(f"Qrels made unreachable by the filter: {lost}/{total} ({lost / total:.1%}) "
          f"-- citations to later-filed documents")

    rows = []
    for label, run_path, _prev in PIPELINE:
        if not os.path.exists(run_path):
            print(f"Skipping {label}: {run_path} not found")
            continue
        run = load_run(run_path)
        filtered = filter_run(run, doc_dates)
        before = aggregate(evaluate_run(qrels, run))
        after = aggregate(evaluate_run(qrels, filtered))
        sig = paired_test_runs(run, filtered, qrels, metric="ndcg_cut_10")
        rows.append({
            "Method": label,
            "later_filed_in_top10": later_filed_at_k(run, qrels, doc_dates),
            "NDCG@10": before["ndcg_cut_10"], "NDCG@10_filtered": after["ndcg_cut_10"],
            "NDCG@10_rel_change": after["ndcg_cut_10"] / before["ndcg_cut_10"] - 1 if before["ndcg_cut_10"] else None,
            "P@10": before["P_10"], "P@10_filtered": after["P_10"],
            "R@10": before["recall_10"], "R@10_filtered": after["recall_10"],
            "MRR": before["recip_rank"], "MRR_filtered": after["recip_rank"],
            "p_value": sig["p_value"],
        })

    df = pd.DataFrame(rows)
    df["p_value_holm"] = holm_adjust(df["p_value"].tolist())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(4))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
