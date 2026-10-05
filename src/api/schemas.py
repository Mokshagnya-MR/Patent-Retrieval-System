from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Method = Literal["bm25", "bm25f", "semantic", "hybrid"]
MAX_K = 50


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    k: int = Field(10, ge=1, le=MAX_K)
    method: Method = "hybrid"
    # Prior-art cutoff: only return patents filed on or before this date.
    filed_before: date | None = None
    # Collapse near-duplicate (same-family) results into one representative.
    diversify: bool = True

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchResultItem(BaseModel):
    rank: int
    patent_id: str
    title: str
    abstract: str = ""
    score: float
    filing_date: str | None = None
    cpc_class: str | None = None  # "H01M" | "G06N" (corpus class)
    main_cpc_label: str | None = None
    highlighted_passages: list[str] = []
    # Near-duplicate patents collapsed into this result by diversification.
    family_members: list[str] = []


class SearchResponse(BaseModel):
    query: str
    method: Method
    alpha: float | None = None  # BM25 weight in hybrid fusion
    filed_before: str | None = None
    n_excluded_later_filed: int = 0
    n_collapsed_duplicates: int = 0
    latency_ms: float
    # Content terms of the query (stopwords removed), for UI highlighting.
    query_terms: list[str] = []
    results: list[SearchResultItem]


class PatentMetadata(BaseModel):
    patent_id: str
    title: str
    abstract: str
    first_claim: str = ""
    n_claims: int = 0
    filing_date: str | None
    publication_date: str | None
    cpc_class: str | None = None
    main_cpc_label: str | None = None
    cpc_codes: list[str]
    inventors: list[dict] = []


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    n_patents: int
    alpha: float
    missing: list[str] = []
