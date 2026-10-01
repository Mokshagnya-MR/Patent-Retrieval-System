"""P@K, R@K, MRR, NDCG@K via pytrec_eval, for K in {5, 10, 20}.

pytrec_eval's measure *spec* (passed to RelevanceEvaluator) uses dotted
names like "ndcg_cut.10", but the *result* dicts it returns always key by
the underscored form "ndcg_cut_10" -- both forms are handled here so
callers only ever see the underscored keys.
"""
import pytrec_eval

CUTOFFS = (5, 10, 20)

MEASURE_SPEC = set()
for k in CUTOFFS:
    MEASURE_SPEC.add(f"ndcg_cut.{k}")
    MEASURE_SPEC.add(f"P.{k}")
    MEASURE_SPEC.add(f"recall.{k}")
MEASURE_SPEC.add("recip_rank")

MEASURES = {m.replace(".", "_") for m in MEASURE_SPEC}


def run_to_pytrec_format(run: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    return {qid: {doc_id: score for doc_id, score in results} for qid, results in run.items()}


def evaluate_run(qrels: dict[str, dict[str, int]], run: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    """Returns {query_id: {measure_name: value}} with underscored measure
    names (e.g. 'ndcg_cut_10'). Only queries present in both qrels and run
    are scored (pytrec_eval's own behavior)."""
    pytrec_run = run_to_pytrec_format(run)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURE_SPEC)
    return evaluator.evaluate(pytrec_run)


def aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {m: 0.0 for m in MEASURES}
    out = {}
    for m in MEASURES:
        vals = [q[m] for q in per_query.values() if m in q]
        out[m] = sum(vals) / len(vals) if vals else 0.0
    return out
