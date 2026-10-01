"""Score every run file in the pipeline against qrels, run paired
significance tests between adjacent pipeline stages, and emit the final
comparison table (both CSV and a markdown block for report/report.md)."""
import argparse
import os

import pandas as pd

from src.eval.metrics import aggregate, evaluate_run
from src.eval.significance import holm_adjust, paired_test
from src.eval.trec_format import load_qrels, load_run

# (label, run_path, "prev" label to test significance against, or None for the first row)
PIPELINE = [
    ("BM25", "results/runs/bm25.run", None),
    ("BM25F", "results/runs/bm25f.run", "BM25"),
    ("Semantic", "results/runs/semantic.run", "BM25F"),
    ("Hybrid (RRF)", "results/runs/hybrid_rrf.run", "Semantic"),
    ("Hybrid (alpha-weighted)", "results/runs/hybrid.run", "Hybrid (RRF)"),
    ("+ Rerank", "results/runs/reranked.run", "Hybrid (alpha-weighted)"),
]


def build_final_table(qrels_path: str, out_csv: str, sig_csv: str, report_md_out: str | None):
    qrels = load_qrels(qrels_path)

    rows = []
    sig_rows = []
    for label, run_path, prev_label in PIPELINE:
        if not os.path.exists(run_path):
            print(f"Skipping {label}: {run_path} not found")
            continue
        run = load_run(run_path)
        avg = aggregate(evaluate_run(qrels, run))

        p_value = None
        if prev_label is not None:
            prev_path = dict((l, p) for l, p, _ in PIPELINE)[prev_label]
            if os.path.exists(prev_path):
                result = paired_test(prev_path, run_path, qrels_path, metric="ndcg_cut_10")
                p_value = result["p_value"]
                sig_rows.append({"comparison": f"{prev_label} vs {label}", **result})

        rows.append({
            "Method": label, "P@10": avg["P_10"], "R@10": avg["recall_10"],
            "MRR": avg["recip_rank"], "NDCG@10": avg["ndcg_cut_10"],
            "vs_prev_p_value": p_value,
        })

    table_df = pd.DataFrame(rows)
    # Five adjacent-stage tests are reported together, so also give the
    # Holm-Bonferroni-adjusted p-values (family-wise error control).
    tested = table_df["vs_prev_p_value"].notna()
    table_df["vs_prev_p_holm"] = None
    if tested.any():
        table_df.loc[tested, "vs_prev_p_holm"] = holm_adjust(table_df.loc[tested, "vs_prev_p_value"].tolist())
    if sig_rows:
        for r, adj in zip(sig_rows, holm_adjust([r["p_value"] for r in sig_rows])):
            r["p_value_holm"] = adj
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(table_df)

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(sig_csv, index=False)
        print(f"Wrote {sig_csv}")

    if report_md_out:
        md_lines = ["| Method | P@10 | R@10 | MRR | NDCG@10 | vs. prev (p) | vs. prev (Holm p) |",
                     "|---|---|---|---|---|---|---|"]
        for _, r in table_df.iterrows():
            p = "—" if pd.isna(r["vs_prev_p_value"]) else f"{r['vs_prev_p_value']:.4f}"
            p_holm = "—" if pd.isna(r["vs_prev_p_holm"]) else f"{r['vs_prev_p_holm']:.4f}"
            md_lines.append(f"| {r['Method']} | {r['P@10']:.4f} | {r['R@10']:.4f} | {r['MRR']:.4f} "
                            f"| {r['NDCG@10']:.4f} | {p} | {p_holm} |")
        print(f"\nMarkdown table (paste into {report_md_out}'s Section 4 by hand -- "
              f"NOT auto-appended, to avoid duplicating the section on re-runs):\n")
        print("\n".join(md_lines))

    return table_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="data/qrels/qrels_test.txt",
                     help="Test-split-only qrels, so every pipeline stage is compared on the same "
                          "held-out queries (the alpha-weighted hybrid run only covers test-split "
                          "queries by design, since alpha was tuned on dev -- scoring other methods "
                          "against the full qrels would compare them on different query counts).")
    ap.add_argument("--out-csv", default="results/final_comparison_table.csv")
    ap.add_argument("--sig-csv", default="results/significance_tests.csv")
    ap.add_argument("--report-md", default="report/report.md")
    args = ap.parse_args()
    build_final_table(args.qrels, args.out_csv, args.sig_csv, args.report_md)


if __name__ == "__main__":
    main()
