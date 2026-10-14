"""Regression tests for bugs found in review (claim splitting on HUPD's
single-line claims, fusion tie reproducibility, diversification families,
highlighting) plus the temporal filter and Holm correction."""
import pandas as pd

from src.eval.significance import holm_adjust
from src.explain.passage_highlight import highlight_lexical, query_terms
from src.retrieve.diversify import dedup_family
from src.retrieve.hybrid import reciprocal_rank_fusion, weighted_fusion
from src.retrieve.query_reduction import extract_claim_1, split_claims
from src.retrieve.temporal import filter_prior_art


# --- HUPD stores all claims on ONE line; claim splitting must work inline ---

def test_split_claims_handles_single_line_hupd_format():
    claims = ("1. A battery comprising an anode. 2. The battery of claim 1, wherein the anode is graphite. "
              "3. A method of making the battery of claim 1.")
    assert split_claims(claims) == [
        "A battery comprising an anode.",
        "The battery of claim 1, wherein the anode is graphite.",
        "A method of making the battery of claim 1.",
    ]


def test_extract_claim_1_skips_canceled_claim_ranges():
    assert extract_claim_1("1-15. (canceled) 16. A quantum circuit. 17. The circuit of claim 16.") == "A quantum circuit."
    assert extract_claim_1("1.-11. (canceled) 12. A system for X. 13. The system of claim 12.") == "A system for X."


def test_extract_claim_1_ignores_in_claim_numbers():
    claims = "1. A cell having 2 electrodes spaced 3.5 mm apart. 2. The cell of claim 1."
    assert extract_claim_1(claims) == "A cell having 2 electrodes spaced 3.5 mm apart."


def test_extract_claim_1_unnumbered_text_falls_back_to_whole_text():
    assert extract_claim_1("A widget.") == "A widget."
    assert extract_claim_1("") == ""


# --- reproducibility: fusion ties must not depend on hash randomization ---

def test_fusion_breaks_ties_by_doc_id():
    fused = weighted_fusion([("b", 1.0), ("a", 1.0)], [], alpha=1.0)
    assert [d for d, _ in fused] == ["a", "b"]
    rrf = reciprocal_rank_fusion([("b", 1.0)], [("a", 1.0)])
    assert [d for d, _ in rrf] == ["a", "b"]


# --- statistics ---

def test_holm_adjust_matches_hand_computation():
    # sorted p: 0.01 * 3 = 0.03, 0.02 * 2 = 0.04, 0.04 * 1 = 0.04 (kept monotone)
    assert holm_adjust([0.04, 0.01, 0.02]) == [0.04, 0.03, 0.04]
    assert holm_adjust([0.5, 0.9]) == [1.0, 1.0]


# --- temporal prior-art filter ---

def test_filter_prior_art_keeps_same_day_and_earlier_only():
    dates = {"old": pd.Timestamp("2014-01-01"), "same": pd.Timestamp("2015-06-05"), "new": pd.Timestamp("2016-01-01")}
    ranked = [("new", 3.0), ("same", 2.0), ("old", 1.0), ("unknown", 0.5)]
    kept = filter_prior_art(ranked, pd.Timestamp("2015-06-05"), dates)
    assert [d for d, _ in kept] == ["same", "old", "unknown"]
    assert filter_prior_art(ranked, None, dates) == ranked


# --- diversification returns family members for the representative ---

def test_dedup_family_groups_near_duplicates_under_top_ranked():
    texts = {"a": "lithium ion battery cathode coating", "b": "lithium ion battery cathode coating",
             "c": "neural network training on graphs"}
    kept, families = dedup_family([("a", 3.0), ("b", 2.0), ("c", 1.0)], texts)
    assert [d for d, _ in kept] == ["a", "c"]
    assert families["a"] == ["b"]


# --- passage highlighting ignores stopwords/patent boilerplate ---

def test_query_terms_drop_stopwords_and_boilerplate():
    assert query_terms("A method wherein the battery comprising a cathode") == {"battery", "cathode"}


def test_highlight_lexical_prefers_content_overlap_over_stopword_overlap():
    doc = "The system of the claim is the one of the said device. A cathode for a lithium battery."
    [(best, score)] = highlight_lexical("lithium battery cathode", doc, top_k=1)
    assert "cathode" in best and score > 0
