"""Pure unit tests for logic that doesn't require the built corpus/indices."""
from src.eval.metrics import aggregate, evaluate_run
from src.index.bm25_index import tokenize
from src.ingest.filter_cpc import _cpc_matches, _split_cpc_codes
from src.preprocess.clean_text import clean_field, strip_claim_numbering, strip_section_markers
from src.retrieve.hybrid import reciprocal_rank_fusion, weighted_fusion
from src.retrieve.query_reduction import extract_claim_1


def test_tokenize_keeps_alphanumeric_chemical_tokens():
    assert tokenize("LiFePO4 cathode") == ["lifepo4", "cathode"]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Cu(NO3)2, battery.") == ["cu", "no3", "2", "battery"]


def test_split_cpc_codes_handles_comma_separated_metadata_format():
    assert _split_cpc_codes("G06F95011, G06F94881") == ["G06F95011", "G06F94881"]


def test_split_cpc_codes_handles_python_list_repr_format():
    assert _split_cpc_codes("['A61B100082', 'A61B100048']") == ["A61B100082", "A61B100048"]


def test_cpc_matches_h01m_and_g06n_prefixes():
    assert _cpc_matches("H01M100066, A61B100") == "H01M"
    assert _cpc_matches("G06N30454") == "G06N"
    assert _cpc_matches("A61B100, H02J700") is None


def test_strip_section_markers_removes_soh_eoh():
    text = "<SOH> BACKGROUND OF THE INVENTION <EOH>This invention relates to batteries."
    assert "SOH" not in strip_section_markers(text)
    assert "batteries" in strip_section_markers(text)


def test_strip_claim_numbering_removes_leading_number_only():
    text = "1. A method comprising 2 steps."
    assert strip_claim_numbering(text) == "A method comprising 2 steps."


def test_clean_field_normalizes_whitespace():
    assert clean_field("  a   b\n\nc  ") == "a b c"


def test_extract_claim_1_returns_first_claim_only():
    claims = "1. A widget comprising a base.\n2. The widget of claim 1, further comprising a lid."
    result = extract_claim_1(claims)
    assert "widget comprising a base" in result
    assert "claim 1" not in result


def test_reciprocal_rank_fusion_favors_docs_ranked_high_in_both_lists():
    ranking_a = [("d1", 5.0), ("d2", 4.0), ("d3", 3.0)]
    ranking_b = [("d2", 0.9), ("d1", 0.5), ("d4", 0.4)]
    fused = reciprocal_rank_fusion(ranking_a, ranking_b)
    fused_ids = [d for d, _ in fused]
    assert fused_ids[0] in ("d1", "d2")
    assert "d3" in fused_ids and "d4" in fused_ids


def test_weighted_fusion_alpha_one_equals_bm25_only_ranking():
    bm25 = [("d1", 10.0), ("d2", 1.0)]
    semantic = [("d2", 0.9), ("d1", 0.1)]
    fused = weighted_fusion(bm25, semantic, alpha=1.0)
    assert fused[0][0] == "d1"


def test_weighted_fusion_alpha_zero_equals_semantic_only_ranking():
    bm25 = [("d1", 10.0), ("d2", 1.0)]
    semantic = [("d2", 0.9), ("d1", 0.1)]
    fused = weighted_fusion(bm25, semantic, alpha=0.0)
    assert fused[0][0] == "d2"


def test_evaluate_run_perfect_ranking_gets_ndcg_one():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": [("d1", 2.0), ("d2", 1.0), ("d3", 0.5)]}
    per_query = evaluate_run(qrels, run)
    assert per_query["q1"]["ndcg_cut_10"] == 1.0


def test_aggregate_averages_across_queries():
    per_query = {"q1": {"ndcg_cut_10": 1.0}, "q2": {"ndcg_cut_10": 0.0}}
    avg = aggregate(per_query)
    assert avg["ndcg_cut_10"] == 0.5
