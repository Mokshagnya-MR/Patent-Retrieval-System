"""API tests against a tiny in-memory corpus: retrieval is stubbed so these
run without the built indices or the embedding model."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ["PATENT_IR_SKIP_WARMUP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

import src.api.main as api  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    subset = pd.DataFrame({
        "patent_id": ["100", "200", "300", "400"],
        "title": ["LITHIUM BATTERY CATHODE", "Lithium battery cathode", "Neural network chip", "Solid electrolyte"],
        "abstract": ["A cathode for a lithium battery.", "A cathode for a lithium battery.",
                     "A chip that runs neural networks.", "A sulfide solid electrolyte."],
        "claims": ["1. A cathode comprising lithium. 2. The cathode of claim 1.", "1. A lithium cathode.",
                   "1-3. (canceled) 4. A neural chip.", "1. A solid electrolyte."],
        "filing_date": pd.to_datetime(["2014-01-01", "2014-02-01", "2016-01-01", "2013-01-01"]),
        "publication_date": pd.to_datetime(["2015-01-01", "2015-02-01", "2017-01-01", "2014-01-01"]),
        "cpc_codes": [np.array(["H01M4131"]), np.array(["H01M4131"]), np.array(["G06N3063"]),
                      np.array(["H01M10056"])],
        "main_cpc_label": ["H01M4131", "H01M4131", "G06N3063", "H01M10056"],
        "inventors": [np.array([{"inventor_name_first": "ADA", "inventor_name_last": "LOVELACE"}]),
                      np.array([], dtype=object), np.array([], dtype=object), np.array([], dtype=object)],
    }).set_index("patent_id", drop=False)
    ranked = [("300", 0.9), ("100", 0.8), ("200", 0.7), ("400", 0.6)]

    monkeypatch.setattr(api, "_subset", subset)
    api.get_doc_dates.cache_clear()
    api.get_alpha.cache_clear()
    monkeypatch.setattr(api, "_retrieve", lambda query, method, n: ranked[:n])
    monkeypatch.setattr(api, "get_bm25_idf", lambda path: {})
    yield TestClient(api.app)
    api.get_doc_dates.cache_clear()
    api.get_alpha.cache_clear()


def test_search_groups_family_members_instead_of_dropping_them(client):
    body = client.post("/search", json={"query": "lithium battery cathode", "k": 10}).json()
    assert [r["patent_id"] for r in body["results"]] == ["300", "100", "400"]
    assert body["results"][1]["family_members"] == ["200"]
    assert body["n_collapsed_duplicates"] == 1
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]


def test_search_without_diversify_returns_all(client):
    body = client.post("/search", json={"query": "lithium", "k": 10, "diversify": False}).json()
    assert [r["patent_id"] for r in body["results"]] == ["300", "100", "200", "400"]


def test_search_filed_before_excludes_later_patents(client):
    body = client.post("/search", json={"query": "lithium", "k": 10, "filed_before": "2014-01-15"}).json()
    assert [r["patent_id"] for r in body["results"]] == ["100", "400"]
    # "300" and "200" are both filed after the cutoff and ranked above "400", the last shown result
    assert body["n_excluded_later_filed"] == 2


def test_similar_excludes_self_and_later_filings_by_default(client):
    body = client.post("/similar/200").json()
    ids = [r["patent_id"] for r in body["results"]]
    assert "200" not in ids and "300" not in ids
    assert body["filed_before"] == "2014-02-01"


def test_similar_unknown_patent_is_404(client):
    assert client.post("/similar/999").status_code == 404


@pytest.mark.parametrize("payload", [{"query": "   "}, {"query": "x", "k": 0}, {"query": "x", "k": 51},
                                     {"query": "x", "method": "tfidf"}])
def test_search_rejects_invalid_input(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_patent_metadata_extracts_first_non_canceled_claim(client):
    body = client.get("/patent/300").json()
    assert body["first_claim"] == "A neural chip."
    assert body["cpc_class"] == "G06N"
    assert client.get("/patent/100").json()["inventors"][0]["inventor_name_last"] == "LOVELACE"
