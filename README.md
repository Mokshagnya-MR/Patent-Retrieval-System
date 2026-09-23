# Patent Retrieval System

AI-powered prior-art search over USPTO patents — an Information Retrieval
course project. BM25/BM25F, semantic (embedding + FAISS) retrieval, hybrid
fusion, learning-to-rank reranking, and citation-derived evaluation are the
graded core; an extractive/RAG explanation layer is a thin, clearly-labeled
bonus on top. **Not a patentability determination tool** — prior-art
discovery only.

## Scope

- **Corpus:** granted US patents (HUPD, Harvard USPTO Patent Dataset) in two
  CPC classes: `H01M` (batteries/electrochemistry) and `G06N` (AI/ML),
  filed 2012–2018. 14,008 patents (10,176 H01M / 3,833 G06N).
- **Ground truth:** citation-derived qrels. A cited patent is relevant to
  its citing patent's query, restricted to citations where both sides fall
  inside our CPC-filtered corpus. 206 usable queries (≥2 in-subset relevant
  docs each), split 144 dev / 62 test.
- **Two query modes:** free-text invention description, and
  existing-patent-as-query (with query reduction).

See `report/report.md` for full methodology, scope-decision rationale
(including two real-world detours: HUPD's loading script no longer runs
under current `datasets` versions, and USPTO's own citation API now
requires an MFA-gated account, so citations are sourced from Google's
Patents Public Data on BigQuery instead), and results.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or: source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

**BigQuery access** (needed for Phase 2, citation fetching): create a GCP
project, enable the BigQuery API, create a service account with the
**BigQuery Job User** role, download its JSON key, then either set
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` or pass `--credentials`
to `fetch_citations.py`. No dataset-create permission needed — citations
are fetched via a query array parameter, not a scratch table.

## Reproducing the pipeline

Run from the repo root, in order:

```bash
# Phase 1 — build the corpus (downloads ~31GB from HuggingFace, checkpoints per year)
python -m src.ingest.filter_cpc

# Phase 2 — citation-derived qrels
python -m src.preprocess.fetch_citations --project YOUR_GCP_PROJECT_ID
python -m src.preprocess.build_qrels
python -m src.eval.splits

# Phase 3 — BM25 / BM25F baselines
python -m src.index.bm25_index
python -m src.index.bm25f_index
python -m src.retrieve.baseline

# Phase 4 — semantic retrieval
python -m src.index.vector_index
python -m src.retrieve.semantic
python -m src.index.ann_benchmark

# Phase 5 — hybrid fusion
python -m src.retrieve.hybrid
python -m src.retrieve.alpha_sweep

# Phase 6 — query reduction ablation
python -m src.retrieve.query_reduction_ablation

# Phase 7 — reranking
python -m src.retrieve.rerank
python -m src.retrieve.rerank_ltr   # stretch: LambdaMART

# Phase 8 — evaluation
python -m src.eval.run_eval --run results/runs/<name>.run
python -m src.eval.final_comparison

# Phase 10 — diversification
python -m src.retrieve.diversify

# Phase 12 — serve it
uvicorn src.api.main:app --reload
streamlit run ui/streamlit_app.py
```

Each script writes its output under `data/` or `results/` (both gitignored
— see below) so later phases pick up right where the previous one left off.

## Tests

```bash
pytest tests/ -v
```

## Repo layout

```
src/
  ingest/      Phase 1  — HUPD download + CPC filtering
  preprocess/  Phase 2  — text cleaning, citation fetch, qrels/topics
  index/       Phase 3-4 — BM25, BM25F, FAISS vector indices
  retrieve/    Phase 3-7,10 — search, fusion, query reduction, reranking, diversification
  eval/        Phase 8  — metrics, significance tests, comparison tables
  explain/     Phase 11 — extractive passage highlighting
  api/         Phase 12 — FastAPI backend
ui/            Phase 12 — Streamlit frontend
notebooks/     Phase 9  — error analysis
report/        Phase 13 — final write-up
```

`data/` and `results/` are gitignored — they hold the built corpus (~1GB+),
indices, and run files, none of which belong in git. Re-run the pipeline
above to regenerate them locally.
