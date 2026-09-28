"""Query reduction for patent-to-patent mode: don't feed a whole patent
(title+abstract+claims+description, often thousands of words) into
retrieval -- it dilutes the query with boilerplate and off-topic sections.

Two strategies:
  - "simple": claim_1 + abstract only (the most legally/topically dense
    parts of a patent).
  - "tfidf": expand the query with the patent's own top-N TF-IDF-weighted
    terms (computed against the corpus), which tends to surface the
    document's distinctive vocabulary better than raw claim_1 text alone.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from src.index.bm25_index import tokenize
from src.preprocess.clean_text import clean_field

QueryMode = str  # "free_text" | "patent_full" | "patent_reduced"

_CLAIM_SPLIT_RE = re.compile(r"\n\s*\d+\s*\.\s*")


def extract_claim_1(claims_text: str) -> str:
    if not isinstance(claims_text, str) or not claims_text.strip():
        return ""
    parts = _CLAIM_SPLIT_RE.split(claims_text.strip())
    parts = [p for p in parts if p.strip()]
    return parts[0].strip() if parts else claims_text.strip()


def reduce_simple(row: dict) -> str:
    """claim_1 + abstract."""
    claim_1 = extract_claim_1(row.get("claims", ""))
    abstract = clean_field(row.get("abstract", ""))
    return f"{clean_field(claim_1, strip_claims=True)} {abstract}".strip()


def build_full_query(row: dict) -> str:
    """The naive baseline this ablation is meant to beat: title + abstract +
    claims + background concatenated, uncut."""
    fields = ["title", "abstract", "claims", "background"]
    return " ".join(clean_field(row.get(f, "")) for f in fields if row.get(f))


class TfidfQueryReducer:
    """Fit once over the corpus, then reduce any patent-as-query row to its
    top-N TF-IDF terms (used as an expanded bag-of-terms query string)."""

    def __init__(self, corpus_texts: list[str], top_n: int = 40):
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, preprocessor=lambda x: x)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def reduce(self, row: dict) -> str:
        full_text = build_full_query(row)
        vec = self.vectorizer.transform([full_text])
        row_arr = vec.toarray()[0]
        top_idx = row_arr.argsort()[::-1][: self.top_n]
        top_terms = [self.feature_names[i] for i in top_idx if row_arr[i] > 0]
        return " ".join(top_terms)


def build_query(row: dict, mode: QueryMode, tfidf_reducer: TfidfQueryReducer | None = None) -> str:
    if mode == "patent_full":
        return build_full_query(row)
    if mode == "patent_reduced":
        if tfidf_reducer is not None:
            return tfidf_reducer.reduce(row)
        return reduce_simple(row)
    raise ValueError(f"build_query only handles patent_full/patent_reduced; got {mode}")
