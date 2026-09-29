"""Two-stage retrieval: retrieve top-100 with hybrid, then re-rank with a
cross-encoder scoring (query, doc) pairs directly. LTR (LambdaMART) is a
documented stretch in rerank_ltr.py, not required for this to function."""
import argparse

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache: dict = {}


def _get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank(query: str, candidates: list[tuple[str, float]], doc_texts: dict[str, str],
           model_name: str = CROSS_ENCODER_MODEL, top_k: int | None = None) -> list[tuple[str, float]]:
    """candidates: [(doc_id, retrieval_score), ...] from stage 1 (hybrid).
    doc_texts: {doc_id: text} for cross-encoder scoring (title+abstract is
    plenty -- cross-encoders are quadratic in input length and we already
    filtered to the top ~100 candidates)."""
    model = _get_cross_encoder(model_name)
    pairs = [(query, doc_texts.get(doc_id, "")) for doc_id, _ in candidates]
    scores = model.predict(pairs)
    reranked = sorted(zip((d for d, _ in candidates), scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k] if top_k else reranked


def main():
    import pandas as pd

    from src.eval.trec_format import load_run, load_topics, write_run

    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-run", default="results/runs/hybrid.run")
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--top-n-candidates", type=int, default=100)
    ap.add_argument("--out", default="results/runs/reranked.run")
    args = ap.parse_args()

    hybrid_run = load_run(args.hybrid_run)
    topics = load_topics(args.topics)
    subset = pd.read_parquet(args.subset)
    subset["patent_id"] = subset["patent_id"].astype(str)
    doc_texts = {
        row["patent_id"]: f"{row['title']}. {row['abstract']}"
        for _, row in subset.iterrows()
    }

    reranked_run = {}
    for qid, query_text in topics.items():
        candidates = hybrid_run.get(qid, [])[: args.top_n_candidates]
        if not candidates:
            continue
        reranked_run[qid] = rerank(query_text, candidates, doc_texts)

    write_run(reranked_run, args.out, tag="reranked")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
