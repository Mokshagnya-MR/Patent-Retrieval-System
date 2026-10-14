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

# HUPD stores ALL claims on a single line with no newlines, e.g.
#   "1-15. (canceled) 16. A quantum circuit ... 17. The circuit of claim 16 ..."
# so claim boundaries have to be found inline. A candidate boundary is a
# claim number (or a canceled range like "1-15." / "1.-11.") followed by
# ".", ")" or ":" and then a capitalized word or "(". Candidates are only
# accepted when their number is the NEXT expected claim number, which
# rejects in-claim references such as "... of claim 1. 2. The method ..."
# (the "1." there is not the expected next number) and stray quantities.
_CLAIM_BOUNDARY_RE = re.compile(
    r"(?<!\S)(\d{1,3})(?:\s*\.?\s*[-–]\s*(\d{1,3}))?\s*[.):](?=\s*[(A-Z])"
)
_CANCELED_RE = re.compile(r"^\(?\s*(canceled|cancelled|deleted)\s*\)?\.?$", re.IGNORECASE)


def split_claims(claims_text: str) -> list[str]:
    """Split a claims field into individual claim texts, in order, with
    canceled claims dropped. Falls back to [whole text] when no numbering
    is found."""
    if not isinstance(claims_text, str) or not claims_text.strip():
        return []
    text = claims_text.strip()
    boundaries = []  # (match_start, text_start)
    expected = 1
    for m in _CLAIM_BOUNDARY_RE.finditer(text):
        start_num = int(m.group(1))
        end_num = int(m.group(2)) if m.group(2) else start_num
        if start_num != expected or end_num < start_num:
            continue
        boundaries.append((m.start(), m.end()))
        expected = end_num + 1
    if not boundaries:
        return [text]

    claims = []
    for i, (_, body_start) in enumerate(boundaries):
        body_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[body_start:body_end].strip()
        if body and not _CANCELED_RE.match(body):
            claims.append(body)
    return claims


def extract_claim_1(claims_text: str) -> str:
    """The first non-canceled claim (claim 1 in the common case; e.g. claim
    16 when the field starts '1-15. (canceled) 16. ...')."""
    claims = split_claims(claims_text)
    return claims[0] if claims else ""


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
