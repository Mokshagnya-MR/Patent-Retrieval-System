"""Build TREC-format qrels + topics from the citation graph.

A cited patent is treated as relevant to its citing patent's query. Relevance
grade: 2 if the citation was examiner-added (category == 'EXA', i.e. the
patent office itself flagged it as prior art -- a stronger relevance signal),
1 for any other in-subset citation (applicant-submitted, etc.). NOTE: in this
corpus, the observed citation.category values from Google's Patents Public
Data are only 'APP' (722) and 'SEA' (467) -- 'EXA' never appears, so in
practice these qrels come out effectively binary. Documented as-is rather
than re-engineering a grading scheme around data we don't have; see the
report's qrels-limitations section.

Only citations where BOTH the citing and cited patent are inside our
CPC-filtered corpus are usable -- a citation to a patent outside H01M/G06N
tells us nothing we can evaluate retrieval against. fetch_citations.py
already restricts to this intersection, but we re-filter here defensively
since qrels correctness is the foundation of every later evaluation number.

Queries (citing patents) with fewer than MIN_RELEVANT in-subset relevant
docs are dropped. The plan's original bar was 3, but this corpus is
deliberately narrow (2 CPC classes, ~14K patents) and qrels only count
STRICTLY in-subset citations, which produces an inherently sparse citation
graph: 759 queries have >=1 relevant doc, 206 have >=2, only 90 have >=3.
Lowered to 2 to reach a topic set (206 queries) safely above the ~150
usable-queries target while still requiring more than a single relevant doc
per query; documented here rather than silently changed.
"""
import argparse
import os

import pandas as pd

MIN_RELEVANT_DOCS = 2


def build_qrels(subset_path: str, citations_path: str, qrels_out: str, topics_out: str) -> pd.DataFrame:
    subset = pd.read_parquet(subset_path)
    subset["patent_id"] = subset["patent_id"].astype(str)
    valid_ids = set(subset["patent_id"])

    citations = pd.read_parquet(citations_path)
    citations["citing_patent_number"] = citations["citing_patent_number"].astype(str)
    citations["cited_patent_number"] = citations["cited_patent_number"].astype(str)

    citations = citations[
        citations["citing_patent_number"].isin(valid_ids)
        & citations["cited_patent_number"].isin(valid_ids)
        & (citations["citing_patent_number"] != citations["cited_patent_number"])
    ]
    print(f"In-subset citation edges: {len(citations)}")

    citations["relevance"] = (citations["citation_category"] == "EXA").astype(int) + 1
    graded = (
        citations.groupby(["citing_patent_number", "cited_patent_number"])["relevance"]
        .max()
        .reset_index()
    )

    counts = graded.groupby("citing_patent_number").size()
    usable_queries = counts[counts >= MIN_RELEVANT_DOCS].index
    graded = graded[graded["citing_patent_number"].isin(usable_queries)]
    print(f"Usable queries (>= {MIN_RELEVANT_DOCS} in-subset relevant docs): {len(usable_queries)}")

    os.makedirs(os.path.dirname(qrels_out), exist_ok=True)
    with open(qrels_out, "w", encoding="utf-8") as f:
        for _, row in graded.sort_values(["citing_patent_number", "cited_patent_number"]).iterrows():
            f.write(f"{row['citing_patent_number']} 0 {row['cited_patent_number']} {row['relevance']}\n")
    print(f"Wrote qrels to {qrels_out}")

    subset_indexed = subset.set_index("patent_id")
    topics_rows = []
    for qid in usable_queries:
        row = subset_indexed.loc[qid]
        title = row["title"] or ""
        abstract = row["abstract"] or ""
        query_text = f"{title}. {abstract}".strip()
        topics_rows.append((qid, query_text))

    with open(topics_out, "w", encoding="utf-8") as f:
        for qid, text in topics_rows:
            f.write(f"{qid}\t{text}\n")
    print(f"Wrote {len(topics_rows)} topics to {topics_out}")

    print("Sample query/relevant-doc pairs:")
    for qid, _ in topics_rows[:5]:
        rel_docs = graded[graded["citing_patent_number"] == qid]["cited_patent_number"].tolist()
        title = subset_indexed.loc[qid, "title"]
        print(f"  Q={qid} ({title[:80]!r}) -> {len(rel_docs)} relevant docs, e.g. {rel_docs[:3]}")

    return graded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/raw/patents_subset.parquet")
    ap.add_argument("--citations", default="data/qrels/raw_citations.parquet")
    ap.add_argument("--qrels-out", default="data/qrels/qrels.txt")
    ap.add_argument("--topics-out", default="data/qrels/topics.tsv")
    args = ap.parse_args()
    build_qrels(args.subset, args.citations, args.qrels_out, args.topics_out)


if __name__ == "__main__":
    main()
