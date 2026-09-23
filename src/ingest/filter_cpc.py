"""Filter the HUPD metadata to our two target CPC classes, decide which
per-year archives are needed, download them, stream-extract only the matching
per-patent JSON records (rather than fully unpacking each multi-GB tar.gz),
and write data/raw/patents_subset.parquet.

Scope decisions (see build-plan doc):
  - CPC classes: H01M (batteries/electrochemistry), G06N (AI/ML/computing)
  - Only GRANTED patents (decision == ACCEPTED, patent_number present) --
    citation ("References Cited") data is only meaningful for granted
    patents, and Phase 2's qrels depend on it.
  - Filing years 2012-2018: verified against the full metadata index that
    this range captures 16,471 of the 16,494 total H01M+G06N granted patents
    across the whole 2004-2018 dataset (99.9%) -- 2004-2011 contributes only
    ~23 patents combined, not worth downloading 8 extra multi-GB archives for.
  - ~16.5K total, split ~11.3K H01M / ~5.2K G06N (all available in-range
    patents are kept; G06N is the smaller class in USPTO filings generally).

The metadata feather stores cpc_labels as a comma-separated STRING (e.g.
"G06F95011, G06F94881"), NOT a JSON array and NOT a Python-list repr -- that
list-repr format only appears in the per-patent JSON files inside the
tar.gz archives (e.g. "['A61B100082', ...]"). Both are handled here.
"""
import argparse
import ast
import json
import os
import re
import tarfile

import pandas as pd
from tqdm import tqdm

from src.ingest.download_hupd import download_metadata, download_year

TARGET_CPC_PREFIXES = ("H01M", "G06N")


def _split_cpc_codes(cpc_field) -> list[str]:
    """Handle both metadata's comma-separated string and the per-patent JSON's
    Python-list-repr string, returning a clean list of CPC code strings."""
    if not cpc_field or not isinstance(cpc_field, str):
        return []
    tokens = re.split(r"[,\[\]]", cpc_field)
    return [t.strip().strip("'\"") for t in tokens if t.strip().strip("'\"")]


def _parse_stringified_list(value):
    """inventor_list is stored as the STRING repr of a list of dicts."""
    if isinstance(value, str) and value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return value


def _cpc_matches(cpc_field) -> str | None:
    for code in _split_cpc_codes(cpc_field):
        for prefix in TARGET_CPC_PREFIXES:
            if code.upper().startswith(prefix):
                return prefix
    return None


def select_subset(metadata_path: str, per_class_cap: int, year_min: int, year_max: int) -> pd.DataFrame:
    df = pd.read_feather(metadata_path)
    print(f"Loaded metadata: {len(df)} rows, columns: {list(df.columns)}")

    df["application_number"] = df["application_number"].astype(str)
    df = df[(df["filing_date"].dt.year >= year_min) & (df["filing_date"].dt.year <= year_max)]

    df = df[df["decision"] == "ACCEPTED"]
    df = df[df["patent_number"].notna() & (df["patent_number"].astype(str).str.len() > 0)]

    df["_cpc_match"] = df["cpc_labels"].apply(_cpc_matches)
    df = df[df["_cpc_match"].notna()]
    print(f"After CPC filter ({TARGET_CPC_PREFIXES}) + granted-only + year range [{year_min},{year_max}]: {len(df)} rows")
    print(df["_cpc_match"].value_counts())

    parts = []
    for prefix, group in df.groupby("_cpc_match"):
        parts.append(group.sample(n=min(per_class_cap, len(group)), random_state=42))
    subset = pd.concat(parts).drop(columns=["_cpc_match"])
    print(f"Final subset: {len(subset)} rows")
    print(subset["filing_date"].dt.year.value_counts().sort_index())
    return subset


def _checkpoint_path(raw_dir: str, year: int) -> str:
    return os.path.join(raw_dir, "checkpoints", f"{year}.parquet")


def extract_full_text(subset_meta: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """For each needed year's tar.gz, stream through and pull only the JSON
    records whose application_number is in our subset. Deletes each archive
    after processing to keep peak disk usage to one year at a time.

    Checkpointed per year to data/raw/checkpoints/{year}.parquet: a year
    whose checkpoint already exists is loaded from disk instead of
    re-downloaded, so an interrupted run only re-does its current year, not
    every year from scratch."""
    subset_meta = subset_meta.copy()
    subset_meta["filing_year"] = subset_meta["filing_date"].dt.year
    wanted_by_year = {
        int(y): set(g["application_number"].astype(str))
        for y, g in subset_meta.groupby("filing_year")
    }

    os.makedirs(os.path.join(raw_dir, "checkpoints"), exist_ok=True)
    year_dfs = []
    for year, wanted_ids in sorted(wanted_by_year.items()):
        ckpt_path = _checkpoint_path(raw_dir, year)
        if os.path.exists(ckpt_path):
            year_df = pd.read_parquet(ckpt_path)
            print(f"Year {year}: loaded {len(year_df)} records from checkpoint {ckpt_path} (skipping download)")
            year_dfs.append(year_df)
            continue

        archive_path = download_year(year, raw_dir)
        print(f"Scanning {archive_path} for {len(wanted_ids)} matching application numbers ...")
        records = {}
        found = 0
        with tarfile.open(archive_path, mode="r:gz") as tf:
            for member in tqdm(tf, desc=f"{year}.tar.gz"):
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                app_id = os.path.splitext(os.path.basename(member.name))[0]
                if app_id not in wanted_ids:
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                try:
                    rec = json.loads(f.read().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                records[app_id] = rec
                found += 1
                if found >= len(wanted_ids):
                    break
        print(f"  matched {found}/{len(wanted_ids)} for year {year}")
        os.remove(archive_path)
        print(f"  removed {archive_path} to free disk space")

        # The per-patent JSON already has its own 'application_number' field
        # (same value as the dict key we indexed by), so just drop the dict-key
        # index rather than resetting it into a column -- that would collide
        # with the JSON's own column of the same name.
        year_df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)
        year_df.to_parquet(ckpt_path, index=False)
        print(f"  saved checkpoint -> {ckpt_path}")
        year_dfs.append(year_df)

    return pd.concat(year_dfs, ignore_index=True) if year_dfs else pd.DataFrame()


def build_subset_parquet(raw_dir: str, out_path: str, per_class_cap: int, year_min: int, year_max: int):
    metadata_path = download_metadata(raw_dir)
    subset_meta = select_subset(metadata_path, per_class_cap, year_min, year_max)

    full_text_df = extract_full_text(subset_meta, raw_dir)
    full_text_df["application_number"] = full_text_df["application_number"].astype(str)

    merged = subset_meta.merge(full_text_df, on="application_number", how="inner", suffixes=("", "_ft"))
    print(f"After merging metadata with extracted full text: {len(merged)} rows")

    out = pd.DataFrame({
        "patent_id": merged["patent_number"].astype(str),
        "application_number": merged["application_number"].astype(str),
        "patent_number": merged["patent_number"].astype(str),
        "title": merged["invention_title"],
        "abstract": merged["abstract"],
        "claims": merged["claims"],
        "background": merged["background"],
        "filing_date": merged["filing_date"],
        "publication_date": merged["date_application_published"],
        "cpc_codes": merged["cpc_labels"].apply(_split_cpc_codes),
        "main_cpc_label": merged["main_cpc_label"],
        "inventors": merged["inventor_list"].apply(_parse_stringified_list),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} patents to {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--output", default="data/raw/patents_subset.parquet")
    ap.add_argument("--per-class-cap", type=int, default=15000)
    ap.add_argument("--year-min", type=int, default=2012)
    ap.add_argument("--year-max", type=int, default=2018)
    args = ap.parse_args()
    build_subset_parquet(args.raw_dir, args.output, args.per_class_cap, args.year_min, args.year_max)


if __name__ == "__main__":
    main()
