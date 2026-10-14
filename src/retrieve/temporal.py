"""Temporal prior-art constraint: a document can only be prior art for a
patent if it existed before that patent was filed. Plain topical retrieval
ignores this, so roughly a third of every method's top-10 are patents filed
AFTER the query patent -- topically similar, but legally impossible prior art
(see src/eval/temporal_filter.py for the measured effect).

Filing date is used as the cutoff because it is the date available for both
sides of every pair in the corpus. It is an approximation of the true legal
test (publication date of the reference vs. priority date of the query):
continuation/co-pending applications can be cited by an earlier-filed
patent, which is why a small share of citation-derived qrels point to
later-filed documents. Same-day filings are kept (<=) rather than dropped.
"""
import pandas as pd


def filter_prior_art(ranked: list[tuple[str, float]], query_date, doc_dates: dict[str, pd.Timestamp]
                     ) -> list[tuple[str, float]]:
    """Keep only docs filed on or before `query_date`. Docs with an unknown
    filing date are kept (no evidence they're ineligible). A missing
    `query_date` disables the filter."""
    if query_date is None or pd.isna(query_date):
        return list(ranked)
    kept = []
    for doc_id, score in ranked:
        d = doc_dates.get(doc_id)
        if d is None or pd.isna(d) or d <= query_date:
            kept.append((doc_id, score))
    return kept
