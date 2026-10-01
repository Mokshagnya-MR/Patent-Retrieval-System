"""Score a run file against qrels; write per-query and averaged metrics CSV."""
import argparse
import os

import pandas as pd

from src.eval.metrics import aggregate, evaluate_run
from src.eval.trec_format import load_qrels, load_run


def evaluate_run_file(run_path: str, qrels_path: str) -> tuple[pd.DataFrame, dict[str, float]]:
    qrels = load_qrels(qrels_path)
    run = load_run(run_path)

    per_query = evaluate_run(qrels, run)
    per_query_df = pd.DataFrame.from_dict(per_query, orient="index")
    per_query_df.index.name = "query_id"

    avg = aggregate(per_query)
    return per_query_df, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qrels", default="data/qrels/qrels.txt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--name", default=None, help="Label for this run (defaults to run filename stem)")
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.run))[0]
    per_query_df, avg = evaluate_run_file(args.run, args.qrels)

    os.makedirs(args.out_dir, exist_ok=True)
    per_query_path = os.path.join(args.out_dir, f"{name}_per_query.csv")
    per_query_df.to_csv(per_query_path)
    print(f"Wrote per-query metrics to {per_query_path}")

    print(f"\n{name} averaged over {len(per_query_df)} queries:")
    for k, v in sorted(avg.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
