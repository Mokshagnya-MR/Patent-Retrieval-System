"""FastAPI app: free-text search, patent-as-query similarity search, and
patent metadata lookup. Run with: uvicorn src.api.main:app --reload"""
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (MAX_K, HealthResponse, Method, PatentMetadata, SearchRequest, SearchResponse,
                             SearchResultItem)
from src.explain.passage_highlight import highlight_lexical, query_terms
from src.ingest.filter_cpc import TARGET_CPC_PREFIXES
from src.preprocess.clean_text import clean_field
from src.retrieve.baseline import get_bm25_idf, search_bm25, search_bm25f
from src.retrieve.diversify import dedup_family
from src.retrieve.hybrid import weighted_fusion
from src.retrieve.query_reduction import split_claims
from src.retrieve.semantic import search_semantic
from src.retrieve.temporal import filter_prior_art

SUBSET_PATH = "data/raw/patents_subset.parquet"
BM25_INDEX = "data/processed/bm25_index.pkl"
BM25F_INDEX = "data/processed/bm25f_index.pkl"
VECTOR_INDEX_DIR = "data/processed/vector_index"
ALPHA_SWEEP_CSV = "results/alpha_sweep.csv"
DEFAULT_ALPHA = 0.5
# Long queries (a whole abstract in patent-as-query mode) contain dozens of
# content terms; highlighting all of them marks most of every result. Only
# the most distinctive ones (highest corpus IDF) are highlighted.
MAX_HIGHLIGHT_TERMS = 12
# Candidate pool per request. Headroom matters: the prior-art filter drops
# ~1/3 of a typical top-10 and family dedup collapses more, and fusion needs
# enough depth from each side to combine.
MIN_CANDIDATES = 100

_subset: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the corpus, indices and embedding model at startup so the first
    user query doesn't pay ~10s of cold loading. Set PATENT_IR_SKIP_WARMUP=1
    to skip (e.g. for tests)."""
    if not os.environ.get("PATENT_IR_SKIP_WARMUP"):
        get_subset()
        get_doc_dates()
        search_bm25("warmup", 1, BM25_INDEX)
        search_bm25f("warmup", 1, index_path=BM25F_INDEX)
        search_semantic("warmup", 1, VECTOR_INDEX_DIR)
    yield


app = FastAPI(title="Patent IR Search API", lifespan=lifespan)


def get_subset() -> pd.DataFrame:
    global _subset
    if _subset is None:
        df = pd.read_parquet(SUBSET_PATH)
        df["patent_id"] = df["patent_id"].astype(str)
        _subset = df.set_index("patent_id", drop=False)
    return _subset


@lru_cache(maxsize=1)
def get_doc_dates() -> dict[str, pd.Timestamp]:
    subset = get_subset()
    return dict(zip(subset["patent_id"], subset["filing_date"]))


@lru_cache(maxsize=1)
def get_alpha() -> float:
    """BM25 weight for hybrid fusion, as selected on the DEV split by
    src/retrieve/alpha_sweep.py (read once; restart the API to pick up a
    re-run sweep)."""
    if os.path.exists(ALPHA_SWEEP_CSV):
        df = pd.read_csv(ALPHA_SWEEP_CSV)
        return float(df.loc[df["ndcg_10"].idxmax(), "alpha"])
    return DEFAULT_ALPHA


def _corpus_class(cpc_codes) -> str | None:
    for code in cpc_codes:
        for prefix in TARGET_CPC_PREFIXES:
            if str(code).upper().startswith(prefix):
                return prefix
    return None


def _date_str(value) -> str | None:
    return value.date().isoformat() if pd.notna(value) else None


def _retrieve(query: str, method: str, n: int) -> list[tuple[str, float]]:
    if method == "bm25":
        return search_bm25(query, n, BM25_INDEX)
    if method == "bm25f":
        return search_bm25f(query, n, index_path=BM25F_INDEX)
    if method == "semantic":
        return search_semantic(query, n, VECTOR_INDEX_DIR)
    if method == "hybrid":
        return weighted_fusion(search_bm25(query, n, BM25_INDEX),
                               search_semantic(query, n, VECTOR_INDEX_DIR), get_alpha())
    raise HTTPException(400, f"Unknown method: {method}")


def _search(query_text: str, method: str, k: int, cutoff, diversify: bool,
            exclude_id: str | None = None) -> tuple[list[SearchResultItem], int, int]:
    """Retrieve -> drop self-match -> prior-art date filter -> family dedup
    -> top k. Returns (items, n_excluded_later_filed, n_collapsed)."""
    subset = get_subset()
    ranked = [(d, s) for d, s in _retrieve(query_text, method, max(MIN_CANDIDATES, 4 * k))
              if d != exclude_id and d in subset.index]

    eligible = filter_prior_art(ranked, cutoff, get_doc_dates())
    families: dict[str, list[str]] = {}
    if diversify:
        eligible, families = dedup_family(eligible, {
            d: f"{subset.at[d, 'title']}. {subset.at[d, 'abstract']}" for d, _ in eligible
        })
    top = eligible[:k]

    # Count later-filed docs the filter removed from ABOVE the last shown
    # result (i.e. ones that would otherwise have been on screen).
    n_excluded = 0
    if top and cutoff is not None:
        kept_ids = {d for d, _ in filter_prior_art(ranked, cutoff, get_doc_dates())}
        last_pos = next(i for i, (d, _) in enumerate(ranked) if d == top[-1][0])
        n_excluded = sum(1 for d, _ in ranked[:last_pos] if d not in kept_ids)

    terms = display_terms(query_text)
    items = []
    for rank, (doc_id, score) in enumerate(top, start=1):
        row = subset.loc[doc_id]
        abstract = clean_field(row["abstract"])
        # Passages come from the claims (they define what the patent covers);
        # the abstract is already shown in full, so highlighting it again adds
        # nothing. Fall back to the abstract for the rare claim-less record.
        claims_text = " ".join(split_claims(clean_field(row["claims"]))) or abstract
        passages = [p for p, s in highlight_lexical(query_text, claims_text, terms=terms) if s > 0]
        items.append(SearchResultItem(
            rank=rank, patent_id=doc_id, title=clean_field(row["title"]), abstract=abstract,
            score=float(score), filing_date=_date_str(row["filing_date"]),
            cpc_class=_corpus_class(row["cpc_codes"]), main_cpc_label=row["main_cpc_label"] or None,
            highlighted_passages=passages, family_members=families.get(doc_id, []),
        ))
    n_collapsed = sum(len(i.family_members) for i in items)
    return items, n_excluded, n_collapsed


def display_terms(query_text: str) -> set[str]:
    terms = query_terms(query_text)
    if len(terms) <= MAX_HIGHLIGHT_TERMS:
        return terms
    idf = get_bm25_idf(BM25_INDEX)
    return set(sorted(terms, key=lambda t: (-idf.get(t, 0.0), t))[:MAX_HIGHLIGHT_TERMS])


@app.get("/health", response_model=HealthResponse)
def health():
    required = [SUBSET_PATH, BM25_INDEX, BM25F_INDEX, os.path.join(VECTOR_INDEX_DIR, "flat.index")]
    missing = [p for p in required if not os.path.exists(p)]
    n = len(get_subset()) if SUBSET_PATH not in missing else 0
    return HealthResponse(status="degraded" if missing else "ok", n_patents=n, alpha=get_alpha(), missing=missing)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    query_text = req.query
    cutoff = pd.Timestamp(req.filed_before) if req.filed_before else None
    items, n_excluded, n_collapsed = _search(query_text, req.method, req.k, cutoff, req.diversify)
    return SearchResponse(
        query=req.query, method=req.method, alpha=get_alpha() if req.method == "hybrid" else None,
        filed_before=req.filed_before.isoformat() if req.filed_before else None,
        n_excluded_later_filed=n_excluded, n_collapsed_duplicates=n_collapsed,
        latency_ms=(time.perf_counter() - t0) * 1000, query_terms=sorted(display_terms(query_text)),
        results=items,
    )


@app.post("/similar/{patent_id}", response_model=SearchResponse)
def similar(patent_id: str, k: int = Query(10, ge=1, le=MAX_K), method: Method = "hybrid",
            prior_art_only: bool = True, diversify: bool = True):
    """Patent-as-query search. The query is the patent's title + abstract --
    the exact formulation every method was evaluated and alpha was tuned on
    (see report Section 5 for the query-reduction ablation). With
    `prior_art_only`, results are restricted to patents filed on or before
    this patent's filing date."""
    t0 = time.perf_counter()
    subset = get_subset()
    patent_id = patent_id.strip()
    if patent_id not in subset.index:
        raise HTTPException(404, f"Patent {patent_id} is not in the indexed corpus")
    row = subset.loc[patent_id]
    query_text = f"{clean_field(row['title'])}. {clean_field(row['abstract'])}"
    cutoff = row["filing_date"] if prior_art_only else None

    items, n_excluded, n_collapsed = _search(query_text, method, k, cutoff, diversify, exclude_id=patent_id)
    return SearchResponse(
        query=query_text, method=method, alpha=get_alpha() if method == "hybrid" else None,
        filed_before=_date_str(cutoff) if cutoff is not None else None,
        n_excluded_later_filed=n_excluded, n_collapsed_duplicates=n_collapsed,
        latency_ms=(time.perf_counter() - t0) * 1000, query_terms=sorted(display_terms(query_text)),
        results=items,
    )


@app.get("/patent/{patent_id}", response_model=PatentMetadata)
def get_patent(patent_id: str):
    subset = get_subset()
    patent_id = patent_id.strip()
    if patent_id not in subset.index:
        raise HTTPException(404, f"Patent {patent_id} is not in the indexed corpus")
    row = subset.loc[patent_id]
    claims = split_claims(clean_field(row["claims"]))
    return PatentMetadata(
        patent_id=patent_id, title=clean_field(row["title"]), abstract=clean_field(row["abstract"]),
        first_claim=claims[0] if claims else "", n_claims=len(claims),
        filing_date=_date_str(row["filing_date"]),
        publication_date=_date_str(row["publication_date"]),
        cpc_class=_corpus_class(row["cpc_codes"]), main_cpc_label=row["main_cpc_label"] or None,
        # cpc_codes/inventors round-trip through parquet as numpy arrays, not
        # plain lists -- `arr or []` raises on multi-element arrays since
        # numpy arrays don't support simple truthiness (same bug fixed in
        # src/retrieve/rerank_ltr.py's cpc_overlap feature).
        cpc_codes=[str(c) for c in row["cpc_codes"]],
        inventors=[dict(i) for i in row["inventors"]] if row["inventors"] is not None else [],
    )
