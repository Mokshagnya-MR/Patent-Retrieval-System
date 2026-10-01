"""Paired significance testing (Wilcoxon signed-rank) between two runs on a
shared per-query metric, e.g. NDCG@10. Use to test each adjacent pipeline
step: BM25 vs hybrid, hybrid vs reranked, etc."""
import argparse

from scipy.stats import wilcoxon

from src.eval.metrics import evaluate_run
from src.eval.trec_format import load_qrels, load_run


def paired_test(run_a_path: str, run_b_path: str, qrels_path: str, metric: str = "ndcg_cut_10"):
    return paired_test_runs(load_run(run_a_path), load_run(run_b_path), load_qrels(qrels_path), metric)


def paired_test_runs(run_a: dict, run_b: dict, qrels: dict, metric: str = "ndcg_cut_10"):
    """Same as paired_test, for runs already held in memory."""
    per_query_a = evaluate_run(qrels, run_a)
    per_query_b = evaluate_run(qrels, run_b)

    shared_qids = sorted(set(per_query_a) & set(per_query_b))
    if not shared_qids:
        raise ValueError("No shared queries between the two runs' evaluated results")

    vals_a = [per_query_a[q][metric] for q in shared_qids]
    vals_b = [per_query_b[q][metric] for q in shared_qids]

    diffs = [b - a for a, b in zip(vals_a, vals_b)]
    if all(d == 0 for d in diffs):
        return {"metric": metric, "n": len(shared_qids), "mean_a": sum(vals_a) / len(vals_a),
                "mean_b": sum(vals_b) / len(vals_b), "statistic": 0.0, "p_value": 1.0}

    statistic, p_value = wilcoxon(vals_a, vals_b)
    return {
        "metric": metric,
        "n": len(shared_qids),
        "mean_a": sum(vals_a) / len(vals_a),
        "mean_b": sum(vals_b) / len(vals_b),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (controls family-wise error when
    several pairwise tests are reported together). Returns adjusted p-values
    in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--qrels", default="data/qrels/qrels.txt")
    ap.add_argument("--metric", default="ndcg_cut_10")
    args = ap.parse_args()

    result = paired_test(args.run_a, args.run_b, args.qrels, args.metric)
    print(result)


if __name__ == "__main__":
    main()
