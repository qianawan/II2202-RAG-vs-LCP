"""Verify final II2202 results and recreate summary plots from the shared CSV files.

Run from the repository root:
    python analysis/analyze_final.py

This post-experiment analysis utility does not rerun Ollama inference.  It
uses only the final per-response scores and final question-type labels.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures" / "reproduced"
METHODS = ("RAG", "LCP")
LENGTHS = ("short", "medium", "long")
HOPS = ("single-hop", "multi-hop")
RESAMPLES = 4000


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def paired_bootstrap(differences, seed, iterations=RESAMPLES):
    """Mean difference and percentile CI, resampling intact question pairs."""
    n = len(differences)
    rng = random.Random(seed)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return (sum(differences) / n,
            estimates[int(0.025 * iterations)],
            estimates[int(0.975 * iterations)])


def make_plots(results, labels):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    for field, ylabel, filename in (
        ("F1", "Mean F1", "f1_vs_length.png"),
        ("latency_sec", "Mean end-to-end latency (seconds)", "latency_vs_length.png"),
        ("total_tokens", "Mean input + output tokens", "tokens_vs_length.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, marker in (("RAG", "o"), ("LCP", "s")):
            values = [mean([r for r in results
                            if r["method"] == method and r["length_group"] == g], field)
                      for g in LENGTHS]
            ax.plot(LENGTHS, values, marker=marker, label=method)
        ax.set(xlabel="Document length group", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, method in enumerate(METHODS):
        values = [mean([r for r in results if r["method"] == method and
                        labels[r["question_id"]] == hop], "F1") for hop in HOPS]
        ax.bar([j + (i - 0.5) * 0.35 for j in range(len(HOPS))],
               values, width=0.35, label=method)
    ax.set_xticks([0, 1], ["Single-hop", "Multi-hop"])
    ax.set_ylabel("Mean F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "f1_by_question_type.png", dpi=180)
    plt.close(fig)


def main():
    results = read_csv(DATA / "results_300_papers.csv")
    classifications = read_csv(DATA / "rq2_labels_300_papers.csv")
    assert len(results) == 600, "Expected 600 response rows"
    assert len(classifications) == 300, "Expected 300 labeled questions"
    ids = [r["question_id"] for r in classifications]
    assert len(set(ids)) == 300
    labels = {r["question_id"]: r["hop_label"] for r in classifications}
    assert Counter(labels.values()) == {"single-hop": 206, "multi-hop": 32, "ambiguous": 62}
    assert Counter((r["length_group"], r["method"]) for r in results) == {
        (length, method): 100 for length in LENGTHS for method in METHODS
    }
    by_question = defaultdict(dict)
    for row in results:
        qid, method = row["question_id"], row["method"]
        assert qid in labels and method in METHODS and method not in by_question[qid]
        by_question[qid][method] = row
    assert set(by_question) == set(ids)
    assert all(set(pair) == set(METHODS) for pair in by_question.values())
    assert len({r["paper_id"] for r in classifications}) == 300

    print("Final dataset: 300 distinct papers; 300 questions; 600 paired responses")
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        print(f"{method}: mean F1={mean(rows, 'F1'):.3f}, "
              f"latency={mean(rows, 'latency_sec'):.2f}s, "
              f"tokens={mean(rows, 'total_tokens'):.0f}")
    print("\nRQ1: LCP minus RAG, paired bootstrap 95% CI")
    for index, length in enumerate(LENGTHS):
        subset = [qid for qid, pair in by_question.items()
                  if pair["RAG"]["length_group"] == length]
        differences = [float(by_question[qid]["LCP"]["F1"])
                       - float(by_question[qid]["RAG"]["F1"]) for qid in subset]
        point, lo, hi = paired_bootstrap(differences, 2026 + index)
        print(f"{length:8} n={len(subset):3} delta={point:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print("\nRQ2: LCP minus RAG, paired bootstrap 95% CI")
    for index, hop in enumerate(HOPS):
        subset = [qid for qid in ids if labels[qid] == hop]
        differences = [float(by_question[qid]["LCP"]["F1"])
                       - float(by_question[qid]["RAG"]["F1"]) for qid in subset]
        point, lo, hi = paired_bootstrap(differences, 2026 + index)
        print(f"{hop:12} n={len(subset):3} delta={point:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    make_plots(results, labels)
    print("\nSaved figures under figures/reproduced/")


if __name__ == "__main__":
    main()
