"""Fetch USPTO patent-citation edges for our corpus from Google's Patents Public
Data on BigQuery (patents-public-data.patents.publications).

Ground-truth source: front-page "References Cited" data, parsed by Google from
the same USPTO examination record that USPTO's own citation feeds are built
from. We restrict to granted US patents (kind codes B1/B2) on both sides of a
citation, since that's where the References Cited list is authoritative.

Usage:
    python -m src.preprocess.fetch_citations \
        --input data/raw/patents_subset.parquet \
        --output data/qrels/raw_citations.parquet \
        --project YOUR_GCP_PROJECT_ID

Requires GOOGLE_APPLICATION_CREDENTIALS env var (or --credentials) pointing at
a service-account JSON key with BigQuery Job User on the target project.
Patent numbers are passed as a query ARRAY parameter (not a scratch table),
so no dataset-create/write permission is needed -- just the ability to run
query jobs and read the public dataset.
"""
import argparse
import os

import pandas as pd
from google.cloud import bigquery

QUERY = """
WITH subset AS (
  SELECT patent_number FROM UNNEST(@patent_numbers) AS patent_number
),
citing_pubs AS (
  SELECT
    p.publication_number,
    REGEXP_EXTRACT(p.publication_number, r'^US-(\\d+)-') AS citing_patent_number,
    c.publication_number AS cited_publication_number,
    c.category AS citation_category,
    c.type AS citation_type
  FROM `patents-public-data.patents.publications` AS p, UNNEST(p.citation) AS c
  WHERE p.country_code = 'US'
    AND p.kind_code IN ('B1', 'B2')
    AND REGEXP_EXTRACT(p.publication_number, r'^US-(\\d+)-') IN (SELECT patent_number FROM subset)
)
SELECT DISTINCT
  citing_patent_number,
  REGEXP_EXTRACT(cited_publication_number, r'^US-(\\d+)-') AS cited_patent_number,
  citation_category,
  citation_type
FROM citing_pubs
WHERE REGEXP_EXTRACT(cited_publication_number, r'^US-(\\d+)-') IN (SELECT patent_number FROM subset)
  AND REGEXP_EXTRACT(cited_publication_number, r'^US-(\\d+)-') != citing_patent_number
"""


def fetch_citations(input_path: str, output_path: str, project: str, credentials: str | None) -> pd.DataFrame:
    if credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

    df = pd.read_parquet(input_path, columns=["patent_number"])
    df = df.dropna(subset=["patent_number"]).drop_duplicates()
    patent_numbers = df["patent_number"].astype(str).tolist()
    print(f"Loaded {len(patent_numbers)} granted patent numbers from subset")

    client = bigquery.Client(project=project)

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("patent_numbers", "STRING", patent_numbers)]
    )
    print("Running citation query against patents-public-data.patents.publications ...")
    result_df = client.query(QUERY, job_config=job_config).to_dataframe()
    print(f"Fetched {len(result_df)} in-subset citation edges")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"Wrote citation edges to {output_path}")
    return result_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/patents_subset.parquet")
    ap.add_argument("--output", default="data/qrels/raw_citations.parquet")
    ap.add_argument("--project", required=True, help="GCP project ID (for billing/job quota)")
    ap.add_argument("--credentials", default=None, help="Path to service-account JSON key")
    args = ap.parse_args()
    fetch_citations(args.input, args.output, args.project, args.credentials)


if __name__ == "__main__":
    main()
