"""Text normalization for patent title/abstract/claims/full-text fields.

Patent text has quirks generic cleaners mishandle: legal boilerplate
("What is claimed is:", "BRIEF SUMMARY"), claim numbering ("1. A method..."),
XML-ish section markers HUPD leaves in (<SOH>...<EOH>), and chemical/alphanumeric
tokens (e.g. "LiFePO4", "Cu(NO3)2") that a naive alphanumeric-only tokenizer
would mangle. We keep those tokens intact rather than stripping digits.
"""
import re

_SOH_EOH_RE = re.compile(r"<SOH>.*?<EOH>", re.DOTALL)
_CLAIM_NUM_RE = re.compile(r"^\s*\d+\s*\.\s*(?=[A-Za-z])")
_WHITESPACE_RE = re.compile(r"\s+")
_BOILERPLATE_PHRASES = [
    "what is claimed is",
    "i/we claim",
    "having thus described",
    "brief description of the drawings",
]


def strip_section_markers(text: str) -> str:
    """Remove HUPD's <SOH>...<EOH> section-header markers (e.g.
    '<SOH> BACKGROUND OF THE INVENTION <EOH>')."""
    if not text:
        return ""
    return _SOH_EOH_RE.sub(" ", text)


def strip_claim_numbering(text: str) -> str:
    """Remove leading claim numbers like '1. ' without touching numbers that
    are part of the claim's substance (e.g. mid-sentence quantities)."""
    if not text:
        return ""
    lines = text.split("\n")
    return "\n".join(_CLAIM_NUM_RE.sub("", line) for line in lines)


def strip_boilerplate(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    for phrase in _BOILERPLATE_PHRASES:
        idx = lowered.find(phrase)
        if idx != -1:
            text = text[:idx]
            lowered = lowered[:idx]
    return text


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_field(text: str, lowercase: bool = False, strip_claims: bool = False) -> str:
    """Full pipeline for one text field. `strip_claims=True` also removes
    leading claim numbering (use for the `claims` field only)."""
    if not isinstance(text, str):
        return ""
    text = strip_section_markers(text)
    if strip_claims:
        text = strip_claim_numbering(text)
    text = strip_boilerplate(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = text.lower()
    return text


def clean_patent_row(row: dict, lowercase: bool = False) -> dict:
    """Clean the text fields of one patent record (title, abstract, claims)."""
    return {
        **row,
        "title_clean": clean_field(row.get("title", ""), lowercase=lowercase),
        "abstract_clean": clean_field(row.get("abstract", ""), lowercase=lowercase),
        "claims_clean": clean_field(row.get("claims", ""), lowercase=lowercase, strip_claims=True),
    }
