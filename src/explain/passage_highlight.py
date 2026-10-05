"""Extractive 'why retrieved' passage highlighting -- for each retrieved doc,
find the passage(s) with the highest term-overlap/embedding-similarity to the
query. This is itself a real IR technique (passage retrieval), and it alone
satisfies the "why was this retrieved" requirement without any generation.

Scope boundary (see report/report.md): this surfaces prior-art passages for a
human to read. It does not, and must not, make or imply any patentability or
novelty determination.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.index.bm25_index import tokenize

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
# Claim/patent boilerplate that carries no topical signal; without this,
# "a"/"the"/"said"/"wherein" overlap dominates the Jaccard score.
_PATENT_STOP_WORDS = {"said", "wherein", "comprising", "comprises", "claim", "claims", "method", "system",
                      "apparatus", "device", "invention", "embodiment", "embodiments", "plurality", "least",
                      "configured", "based", "using", "include", "includes", "including", "provided"}
STOP_WORDS = ENGLISH_STOP_WORDS | _PATENT_STOP_WORDS


def query_terms(query: str) -> set[str]:
    """Content-bearing query tokens (stopwords and 1-char tokens removed)."""
    return {t for t in tokenize(query) if t not in STOP_WORDS and len(t) > 1}


def split_passages(text: str, max_sentences_per_passage: int = 2) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    passages = []
    for i in range(0, len(sentences), max_sentences_per_passage):
        passages.append(" ".join(sentences[i: i + max_sentences_per_passage]))
    return passages


def highlight_lexical(query: str, doc_text: str, top_k: int = 2,
                      terms: set[str] | None = None) -> list[tuple[str, float]]:
    """Score each passage by content-token overlap (Jaccard) with the query --
    cheap, interpretable, no model call. Pass precomputed `terms` (from
    query_terms) to avoid re-tokenizing the query per document."""
    query_tokens = terms if terms is not None else query_terms(query)
    if not query_tokens:
        return []
    passages = split_passages(doc_text)
    scored = []
    for p in passages:
        p_tokens = {t for t in tokenize(p) if t not in STOP_WORDS and len(t) > 1}
        if not p_tokens:
            continue
        overlap = len(query_tokens & p_tokens) / len(query_tokens | p_tokens)
        scored.append((p, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def highlight_semantic(query: str, doc_text: str, model, top_k: int = 2) -> list[tuple[str, float]]:
    """Score each passage by cosine similarity to the query embedding --
    catches paraphrase/vocabulary-mismatch cases lexical overlap misses."""
    passages = split_passages(doc_text)
    if not passages:
        return []
    embeddings = model.encode([query] + passages, normalize_embeddings=True, convert_to_numpy=True)
    query_emb, passage_embs = embeddings[0], embeddings[1:]
    scores = passage_embs @ query_emb
    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
