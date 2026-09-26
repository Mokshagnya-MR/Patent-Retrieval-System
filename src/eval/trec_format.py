"""Shared TREC-format I/O: topics, qrels, and run files."""
import os


def load_topics(path: str) -> dict[str, str]:
    topics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            topics[qid] = text
    return topics


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def write_run(run: dict[str, list[tuple[str, float]]], path: str, tag: str):
    """run: {query_id: [(doc_id, score), ...]} already sorted best-first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, results in run.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def load_run(path: str) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, list[tuple[str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            qid, _, doc_id, _rank, score, _tag = parts
            run.setdefault(qid, []).append((doc_id, float(score)))
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run
