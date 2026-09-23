"""Download raw HUPD files directly from the HF Hub repo (bypassing the
`datasets` library loading script, which HF's `datasets>=4` no longer runs).

The HUPD/hupd repo stores:
  - hupd_metadata_2022-02-22.feather : bibliographic metadata for ALL ~4.5M
    patents (title, dates, CPC/IPC labels, decision, application_number,
    patent_number). Small enough to filter locally.
  - data/{year}.tar.gz : per-year archives of per-patent JSON files (full
    text: abstract, claims, background, summary, full_description), keyed by
    application_number.

We download the metadata file once, then only the per-year tar.gz archives
that filter_cpc.py determines are actually needed.
"""
import argparse
import os
import time

import requests
from tqdm import tqdm

HF_BASE = "https://huggingface.co/datasets/HUPD/hupd/resolve/main"
METADATA_FILENAME = "hupd_metadata_2022-02-22.feather"

MAX_RETRIES = 8
RETRY_BACKOFF_CAP_SECONDS = 60


def _download(url: str, dest: str, max_retries: int = MAX_RETRIES):
    """Streams to a .part file, resuming via HTTP Range on connection drops
    (these multi-GB archives take 20-30+ min each over a single HTTP stream,
    long enough that a dropped connection is a routine event, not an
    exception). Falls back to a full restart if the server doesn't honor the
    Range request."""
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    for attempt in range(1, max_retries + 1):
        resume_from = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if resume_from > 0 and r.status_code == 206:
                    mode, initial = "ab", resume_from
                else:
                    r.raise_for_status()
                    mode, initial = "wb", 0
                total = int(r.headers.get("content-length", 0)) + initial
                with open(tmp, mode) as f, tqdm(total=total, initial=initial, unit="B", unit_scale=True,
                                                 desc=os.path.basename(dest)) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            os.replace(tmp, dest)
            return
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, RETRY_BACKOFF_CAP_SECONDS)
            print(f"\nDownload error (attempt {attempt}/{max_retries}): {e}\nRetrying in {wait}s, "
                  f"resuming from {os.path.getsize(tmp) if os.path.exists(tmp) else 0} bytes ...")
            time.sleep(wait)


def download_metadata(raw_dir: str) -> str:
    dest = os.path.join(raw_dir, METADATA_FILENAME)
    _download(f"{HF_BASE}/{METADATA_FILENAME}", dest)
    return dest


def download_year(year: int, raw_dir: str) -> str:
    dest = os.path.join(raw_dir, f"{year}.tar.gz")
    _download(f"{HF_BASE}/data/{year}.tar.gz", dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--years", nargs="*", type=int, default=[],
                     help="Per-year tar.gz archives to download (in addition to metadata)")
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    download_metadata(args.raw_dir)
    if not args.metadata_only:
        for y in args.years:
            download_year(y, args.raw_dir)


if __name__ == "__main__":
    main()
