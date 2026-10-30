
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUB = DATA / "rq2_submission"
OUT = DATA / "rq1_300_distinct_papers"

OLD = SUB / "rq2_labels.csv"
NEW = SUB / "rq2_review_300_papers_annotated.csv"
SAMPLES = OUT / "questions_300_papers.jsonl"
RESULTS = OUT / "results_300_papers.csv"

VALID = {"single-hop", "multi-hop", "ambiguous"}

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def normalize(value):
    value = str(value or "").strip().lower()
    value = value.replace("_", "-").replace(" ", "-")
    return value

samples = [
    json.loads(line)
    for line in SAMPLES.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
sample_ids = {str(s["question_id"]) for s in samples}
assert len(samples) == len(sample_ids) == 300

old_rows = load_csv(OLD)
new_rows = load_csv(NEW)

labels = {}

# Preserve the 105 previously classified questions.
for row in old_rows:
    qid = row["question_id"].strip()
    label = normalize(row.get("hop_label", ""))
    if label in ("single-hop", "multi-hop"):
        if qid in labels:
            raise ValueError(f"Duplicate old label: {qid}")
        labels[qid] = (label, row.get("reason", ""))

old_count = len(labels)

# Add the 195 newly reviewed questions.
new_seen = set()
for row in new_rows:
    qid = row["question_id"].strip()
    label = normalize(row.get("final_label", ""))
    if qid in new_seen:
        raise ValueError(f"Duplicate new question: {qid}")
    new_seen.add(qid)

    if label not in VALID:
        raise ValueError(f"Missing or invalid final_label for {qid}: {label!r}")
    if qid in labels:
        raise ValueError(f"Overlapping old/new question: {qid}")

    labels[qid] = (label, row.get("final_reason", ""))

print("Old classified:", old_count)
print("New reviewed:", len(new_seen))
print("Merged:", len(labels))
print("Label counts:", dict(Counter(x[0] for x in labels.values())))

missing = sample_ids - set(labels)
extra = set(labels) - sample_ids

if missing or extra or len(labels) != 300:
    raise RuntimeError(
        f"Question ID mismatch! Missing={len(missing)}, Extra={len(extra)}\n"
        f"Missing examples: {list(missing)[:5]}\n"
        f"Extra examples: {list(extra)[:5]}"
    )

# Save final labels without overwriting original files.
label_path = OUT / "rq2_labels_300_papers.csv"
with label_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "question_id", "paper_id", "length_group",
            "hop_label", "reason"
        ]
    )
    writer.writeheader()
    for s in samples:
        qid = str(s["question_id"])
        label, reason = labels[qid]
        writer.writerow({
            "question_id": qid,
            "paper_id": s["paper_id"],
            "length_group": s["length_group"],
            "hop_label": label,
            "reason": reason,
        })

# Load actual model outputs from the final 300-paper experiment.
results = load_csv(RESULTS)
pairs = {(r["question_id"], r["method"]) for r in results}
required = {
    (qid, method)
    for qid in sample_ids
    for method in ("RAG", "LCP")
}

if len(results) != 600 or pairs != required:
    raise RuntimeError("Final model results do not match the 300 questions.")

summary = []

for hop in ("single-hop", "multi-hop"):
    ids = {qid for qid, (label, _) in labels.items() if label == hop}

    for method in ("RAG", "LCP"):
        rows = [
            r for r in results
            if r["question_id"] in ids and r["method"] == method
        ]
        if not rows:
            continue

        summary.append({
            "question_type": hop,
            "method": method,
            "n": len(rows),
            "EM": sum(float(r["EM"]) for r in rows) / len(rows),
            "F1": sum(float(r["F1"]) for r in rows) / len(rows),
        })

summary_path = OUT / "rq2_summary_300_papers.csv"
with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "question_type", "method", "n", "EM", "F1"
    ])
    writer.writeheader()
    writer.writerows(summary)

for row in summary:
    print(
        f"{row['question_type']:12} {row['method']:3} "
        f"n={row['n']:3} EM={row['EM']:.3f} F1={row['F1']:.3f}"
    )

# Plot real RQ2 F1 and EM.
for metric in ("F1", "EM"):
    groups = ("single-hop", "multi-hop")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, method in enumerate(("RAG", "LCP")):
        xs = [i + (idx - 0.5) * 0.35 for i in range(2)]
        values = []
        for hop in groups:
            match = [
                r[metric] for r in summary
                if r["question_type"] == hop and r["method"] == method
            ]
            values.append(match[0] if match else 0)
        ax.bar(xs, values, width=0.35, label=method)

    ax.set_xticks([0, 1], ["Single-hop", "Multi-hop"])
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"{metric} by Question Type")
    ax.legend()
    fig.tight_layout()

    name = f"{metric.lower()}_by_question_type_300_papers.png"
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)

print("\nSUCCESS: All 300 question IDs matched.")
print("Final labels:", label_path)
print("RQ2 summary:", summary_path)
print("RQ2 F1 and EM charts:", OUT)
