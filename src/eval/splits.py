"""Deterministic 70/30 dev/test split over topic (query) ids, persisted so
every phase that needs it (alpha tuning, query-reduction ablation, LTR
training) uses the exact same split. Tune only on dev; report only on test."""
import argparse
import os
import random

DEV_FRACTION = 0.7


def make_split(topics: dict[str, str], seed: int = 42) -> tuple[list[str], list[str]]:
    qids = sorted(topics.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_dev = int(len(qids) * DEV_FRACTION)
    return qids[:n_dev], qids[n_dev:]


def save_split(dev_ids: list[str], test_ids: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dev_topics.txt"), "w") as f:
        f.write("\n".join(dev_ids) + "\n")
    with open(os.path.join(out_dir, "test_topics.txt"), "w") as f:
        f.write("\n".join(test_ids) + "\n")


def load_split(split_dir: str) -> tuple[list[str], list[str]]:
    with open(os.path.join(split_dir, "dev_topics.txt")) as f:
        dev_ids = [line.strip() for line in f if line.strip()]
    with open(os.path.join(split_dir, "test_topics.txt")) as f:
        test_ids = [line.strip() for line in f if line.strip()]
    return dev_ids, test_ids


def main():
    from src.eval.trec_format import load_topics

    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="data/qrels/topics.tsv")
    ap.add_argument("--out-dir", default="data/qrels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topics = load_topics(args.topics)
    dev_ids, test_ids = make_split(topics, args.seed)
    save_split(dev_ids, test_ids, args.out_dir)
    print(f"Dev: {len(dev_ids)} topics, Test: {len(test_ids)} topics -> {args.out_dir}")


if __name__ == "__main__":
    main()
