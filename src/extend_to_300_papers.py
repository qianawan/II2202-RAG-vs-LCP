r"""Extend the II2202 QASPER study to 300 questions from 300 distinct papers.

Keeps the ORIGINAL 272 dev-paper question/response pairs (544 responses) and
samples 28 NEW papers from the QASPER v0.3 TRAIN split, using the SAME
eligibility rules, length boundaries, model, retrieval and scoring settings.

Windows PowerShell commands (run beside experiment.py):
  .\.venv\Scripts\python.exe extend_to_300_papers.py prepare
  .\.venv\Scripts\python.exe extend_to_300_papers.py run
  .\.venv\Scripts\python.exe extend_to_300_papers.py summarize
  .\.venv\Scripts\python.exe extend_to_300_papers.py pack-rq2

Never modifies original samples/results or data/rq2_labels.csv. The previous
300-question / 272-paper data under data/rq1_to_300 stays untouched but is NOT
included in this new analysis.
"""

import argparse
import csv
import json
import random
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import experiment as base

DATA = base.DATA
OUT = DATA / "rq1_300_distinct_papers"
TRAIN_SAMPLES = OUT / "train_28_questions.jsonl"
PARTIAL = OUT / "pilot_results.jsonl"  # experiment.run() for limit<150
TRAIN_RESULTS = OUT / "train_28_results.jsonl"
FINAL_SAMPLES = OUT / "questions_300_papers.jsonl"
FINAL_CSV = OUT / "results_300_papers.csv"
GROUPS = ("short", "medium", "long")
TARGETS = {"short": 9, "medium": 10, "long": 9}
ORIGINAL_SAMPLE_PATHS = (
    DATA / "qasper_150.jsonl",
    DATA / "rq1_extension" / "extra_75_questions.jsonl",
    DATA / "rq1_remaining" / "remaining_questions.jsonl",
)
ORIGINAL_RESULT_PATHS = (
    DATA / "real_results.jsonl",
    DATA / "rq1_extension" / "extra_75_results.jsonl",
    DATA / "rq1_remaining" / "remaining_results.jsonl",
)


def read_many(paths):
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path}")
        rows.extend(base.read_jsonl(path))
    return rows


def expected_config(ctx):
    return {
        "model": base.MODEL, "num_ctx": ctx,
        "num_predict": base.NUM_PREDICT, "temperature": 0,
        "seed": base.SEED, "top_k": base.TOP_K,
        "chunk_words": base.CHUNK_WORDS,
        "overlap_words": base.CHUNK_OVERLAP_WORDS,
    }


def validate_old():
    samples = read_many(ORIGINAL_SAMPLE_PATHS)
    results = read_many(ORIGINAL_RESULT_PATHS)
    if len(samples) != 272 or len(results) != 544:
        raise RuntimeError("Expected completed 272-paper study (272 questions, 544 responses).")
    qids = [q["question_id"] for q in samples]
    pids = [q["paper_id"] for q in samples]
    if len(set(qids)) != 272 or len(set(pids)) != 272:
        raise RuntimeError("Original 272 samples must be from 272 distinct papers.")
    if Counter(q["length_group"] for q in samples) != {"short": 91, "medium": 90, "long": 91}:
        raise RuntimeError("Unexpected original document-length distribution.")
    pairs = {(r["question_id"], r["method"]) for r in results}
    if len(pairs) != 544 or pairs != {(qid, m) for qid in qids for m in ("RAG", "LCP")}:
        raise RuntimeError("Original results contain missing/duplicate question-method pairs.")
    config = results[0]["config"]
    if any(r["config"] != config for r in results):
        raise RuntimeError("Original experiments do not share the same model configuration.")
    return samples, results, config


def eligible_qas(paper):
    """Mirror experiment.prepare eligibility criteria without changing references."""
    choices = []
    for qa in paper.get("qas", []):
        answers, evidence = [], []
        for annotation in qa.get("answers", []):
            obj = annotation.get("answer") or {}
            ans = base.annotated_answer(obj)
            ev = [str(e).strip() for e in (obj.get("evidence") or []) if str(e).strip()]
            text_ev = [e for e in ev if not e.upper().startswith("FLOAT SELECTED")]
            if ans and text_ev:
                answers.append(ans)
                evidence.extend(text_ev)
        if answers and evidence:
            choices.append({
                "question_id": str(qa["question_id"]),
                "question": qa["question"],
                "references": list(dict.fromkeys(answers)),
                "evidence": list(dict.fromkeys(evidence)),
            })
    return choices


def load_train_split():
    if not base.ARCHIVE.is_file():
        print("Official QASPER archive missing; downloading it using experiment.py...")
        base.download_qasper()
    with tarfile.open(base.ARCHIVE, "r:gz") as archive:
        matches = [m for m in archive.getmembers()
                   if m.isfile() and m.name.endswith("qasper-train-v0.3.json")]
        if len(matches) != 1:
            raise RuntimeError("Could not uniquely identify qasper-train-v0.3.json "
                               "inside the official QASPER archive. Found: "
                               + ", ".join(m.name for m in archive.getmembers() if m.isfile()))
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise RuntimeError("Could not read QASPER train split.")
        return json.load(stream)


def prepare():
    old, _, _ = validate_old()
    OUT.mkdir(parents=True, exist_ok=True)
    if TRAIN_SAMPLES.exists():
        saved = base.read_jsonl(TRAIN_SAMPLES)
        if len(saved) != 28:
            raise RuntimeError("Existing train-sample file has unexpected length. Do not overwrite it.")
        print("Existing 28 train-split questions retained:", TRAIN_SAMPLES)
        return

    meta = json.loads((DATA / "sampling_metadata.json").read_text(encoding="utf-8"))
    lo, hi = meta["short_upper"], meta["medium_upper"]
    max_length = meta["max_doc_proxy_tokens"]
    existing_papers = {r["paper_id"] for r in old}
    existing_questions = {r["question_id"] for r in old}
    existing_titles = {" ".join(r.get("title", "").lower().split()) for r in old}
    train = load_train_split()
    candidates = {g: [] for g in GROUPS}

    for pid, paper in sorted(train.items()):
        if pid in existing_papers:
            continue
        title = " ".join(paper.get("title", "").lower().split())
        if title and title in existing_titles:
            continue
        text = base.paper_text(paper)
        if not text:
            continue
        tokens = len(base.ENCODER.encode(text))
        if tokens > max_length:
            continue
        group = "short" if tokens <= lo else "medium" if tokens <= hi else "long"
        choices = [q for q in eligible_qas(paper) if q["question_id"] not in existing_questions]
        if not choices:
            continue
        choices.sort(key=lambda q: q["question_id"])
        question = random.Random(f"{base.SEED}:{pid}").choice(choices)
        candidates[group].append({
            "paper_id": pid, "title": paper.get("title", ""),
            "paper_text": text, "proxy_tokens": tokens,
            "length_group": group, **question,
        })

    print("Additional eligible TRAIN papers, by original length boundaries:")
    for group in GROUPS:
        print(f"  {group}: {len(candidates[group])} available; need {TARGETS[group]}")
        if len(candidates[group]) < TARGETS[group]:
            raise RuntimeError("Insufficient eligible new train papers. No sample file was written.")

    rng = random.Random(base.SEED)
    selected = {}
    for group in GROUPS:
        rng.shuffle(candidates[group])
        selected[group] = candidates[group][:TARGETS[group]]
    extra = [selected[g][i] for i in range(max(TARGETS.values()))
             for g in GROUPS if i < len(selected[g])]
    final = old + extra
    if not (len(extra) == 28 and len(final) == 300
            and len({r["paper_id"] for r in final}) == 300
            and len({r["question_id"] for r in final}) == 300
            and Counter(r["length_group"] for r in final) == {g: 100 for g in GROUPS}):
        raise RuntimeError("Failed uniqueness/group validation. No sample file was written.")

    base.write_jsonl(TRAIN_SAMPLES, extra)
    (OUT / "sampling_metadata_300_papers.json").write_text(json.dumps({
        "dataset": "QASPER v0.3",
        "original_split": "dev/validation", "new_split": "train",
        "dev_papers": 272, "train_papers": 28,
        "questions": 300, "distinct_papers": 300, "model_responses": 600,
        "length_group_counts": {g: 100 for g in GROUPS},
        "train_group_counts": TARGETS,
        "short_upper": lo, "medium_upper": hi,
        "document_proxy_tokenizer": base.ENCODING,
        "max_doc_proxy_tokens": max_length,
        "random_seed": base.SEED,
        "sampling_rule": "fixed-seed random one-question-per-new-paper from eligible QASPER train papers",
        "reason_for_extension": "extend 272 dev papers to exactly 300 distinct papers; exploratory split extension",
    }, indent=2), encoding="utf-8")
    print("Saved:", TRAIN_SAMPLES)
    print("All original results and RQ2 labels preserved.")


def run():
    _, _, config = validate_old()
    if config != expected_config(config["num_ctx"]):
        raise RuntimeError("Current model/retrieval settings differ from original experiment.")
    if not TRAIN_SAMPLES.is_file():
        raise RuntimeError("Run prepare first.")
    extra = base.read_jsonl(TRAIN_SAMPLES)
    if len(extra) != 28:
        raise RuntimeError("Expected 28 new paper questions.")
    if TRAIN_RESULTS.is_file():
        done = base.read_jsonl(TRAIN_RESULTS)
        required = {(r["question_id"], m) for r in extra for m in ("RAG", "LCP")}
        if len(done) == 56 and {(r["question_id"], r["method"]) for r in done} == required:
            print("Already complete:", TRAIN_RESULTS)
            return
        raise RuntimeError("Finished-file exists but is incomplete; investigate without overwriting.")

    old_data, old_samples = base.DATA, base.SAMPLES
    try:
        base.DATA, base.SAMPLES = OUT, TRAIN_SAMPLES
        base.run(argparse.Namespace(limit=28, ctx=config["num_ctx"]))
    finally:
        base.DATA, base.SAMPLES = old_data, old_samples

    rows = base.read_jsonl(PARTIAL)
    required = {(r["question_id"], m) for r in extra for m in ("RAG", "LCP")}
    if len(rows) == 56 and {(r["question_id"], r["method"]) for r in rows} == required:
        if any(r["config"] != config for r in rows):
            raise RuntimeError("Train inference config differs from original; review results.")
        PARTIAL.rename(TRAIN_RESULTS)
        print("Extension completed:", TRAIN_RESULTS)
    else:
        print("Incomplete: rerun the same command to resume safely.")


def cluster_ci(deltas, seed, iterations=4000):
    rng = random.Random(seed)
    vals = list(deltas)
    n = len(vals)
    estimates = []
    for _ in range(iterations):
        estimates.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]


def get_final():
    old_samples, old_results, config = validate_old()
    if not TRAIN_SAMPLES.is_file():
        raise RuntimeError("Run prepare first.")
    samples = old_samples + base.read_jsonl(TRAIN_SAMPLES)
    if len(samples) != 300 or len({s["paper_id"] for s in samples}) != 300:
        raise RuntimeError("Expected 300 questions from 300 distinct papers.")
    return samples, old_results, config


def summarize():
    samples, old_results, config = get_final()
    if not TRAIN_RESULTS.is_file():
        raise RuntimeError("Run the 28 new questions first.")
    results = old_results + base.read_jsonl(TRAIN_RESULTS)
    if len(results) != 600 or len({(r["question_id"], r["method"]) for r in results}) != 600:
        raise RuntimeError("Expected 600 unique question/method responses.")
    if any(r["config"] != config for r in results):
        raise RuntimeError("Model/retrieval configuration mismatch.")
    pairs = {(r["question_id"], r["method"]) for r in results}
    expected = {(q["question_id"], m) for q in samples for m in ("RAG", "LCP")}
    if pairs != expected or Counter(s["length_group"] for s in samples) != {g: 100 for g in GROUPS}:
        raise RuntimeError("Missing data or incorrect length group allocation.")

    print("\nFINAL 300 QUESTIONS / 300 DISTINCT PAPERS / 600 MODEL RESPONSES")
    print("Data provenance: 272 dev papers + 28 train papers")
    for group in (*GROUPS, "ALL"):
        for method in ("RAG", "LCP"):
            subset = [r for r in results if r["method"] == method and
                      (group == "ALL" or r["length_group"] == group)]
            def avg(field):
                return sum(r[field] for r in subset) / len(subset)
            print(f"{group:6} {method:3} n={len(subset):3} "
                  f"EM={avg('EM'):.3f} F1={avg('F1'):.3f} "
                  f"latency={avg('latency_sec'):.2f}s tokens={avg('total_tokens'):.0f}")

    by_id = {(r["question_id"], r["method"]): r for r in results}
    for metric in ("F1", "EM"):
        print(f"\n{metric}: paired LCP-RAG difference and 95% PAPER-bootstrap CI")
        signs = []
        for idx, group in enumerate(GROUPS):
            ids = [s["question_id"] for s in samples if s["length_group"] == group]
            deltas = [by_id[(qid, "LCP")][metric] - by_id[(qid, "RAG")][metric] for qid in ids]
            observed = sum(deltas) / len(deltas)
            lo, hi = cluster_ci(deltas, 2026 + idx)
            signs.append((observed > 0) - (observed < 0))
            print(f"  {group}: difference={observed:+.4f}; CI=[{lo:+.4f}, {hi:+.4f}]")
        cross = any(signs[i] * signs[i + 1] < 0 for i in (0, 1))
        print(f"  Observed ranking reversal: {cross} (exploratory; not a significance test)")

    base.write_jsonl(FINAL_SAMPLES, samples)
    columns = ["question_id", "paper_id", "length_group", "document_proxy_tokens",
               "method", "prediction", "EM", "F1", "latency_sec",
               "input_tokens", "output_tokens", "total_tokens"]
    with FINAL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    import matplotlib.pyplot as plt
    for field, ylabel, filename in (
        ("F1", "Mean F1", "f1_vs_length_300_papers.png"),
        ("EM", "Mean Exact Match", "em_vs_length_300_papers.png"),
        ("latency_sec", "Mean latency (s)", "latency_vs_length_300_papers.png"),
        ("total_tokens", "Mean input + output tokens", "tokens_vs_length_300_papers.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for method, marker, style in (("RAG", "o", "-"), ("LCP", "s", "--")):
            means = [sum(r[field] for r in results if r["method"] == method
                         and r["length_group"] == g) / 100 for g in GROUPS]
            ax.plot(GROUPS, means, marker=marker, linestyle=style, label=method)
        ax.set(xlabel="Document-length group", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)
    print("\nSaved:", FINAL_CSV)
    print("Four REAL 300-paper charts saved to:", OUT)
    print("Excluded previous dev-duplicate 28 questions and preserved their original files.")


def pack_rq2():
    """Build an UPDATED review ZIP; earlier 300-question/272-paper ZIP is obsolete."""
    samples, _, _ = get_final()
    if not (DATA / "rq2_labels.csv").exists():
        raise FileNotFoundError(DATA / "rq2_labels.csv")
    with (DATA / "rq2_labels.csv").open(encoding="utf-8-sig", newline="") as f:
        old_labels = {r["question_id"]: r for r in csv.DictReader(f)}
    work = OUT / "RQ2_for_teammate_300_papers"
    paper_folder = work / "papers"
    paper_folder.mkdir(parents=True, exist_ok=True)
    rows, count = [], Counter()
    for sample in samples:
        qid = sample["question_id"]
        old = old_labels.get(qid, {})
        label = old.get("hop_label", "").strip().lower()
        if label in ("single-hop", "multi-hop"):
            count["previously_labeled"] += 1
            continue
        count["needs_review"] += 1
        rows.append({
            "question_id": qid, "paper_id": sample["paper_id"],
            "paper_title": sample["title"], "length_group": sample["length_group"],
            "question": sample["question"],
            "reference_answers": json.dumps(sample["references"], ensure_ascii=False),
            "supporting_evidence": "\n\n".join(sample["evidence"]),
            "previous_label": label, "previous_reason": old.get("reason", ""),
            "final_label": "", "final_reason": "", "evidence_location": "",
        })
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", qid)
        (paper_folder / f"{safe_id}.txt").write_text(
            f"TITLE: {sample['title']}\nPAPER ID: {sample['paper_id']}\nQUESTION ID: {qid}\n\n"
            + f"QUESTION: {sample['question']}\n\nREFERENCES: "
            + json.dumps(sample["references"], ensure_ascii=False, indent=2)
            + "\n\nEVIDENCE:\n" + "\n\n".join(sample["evidence"])
            + "\n\nFULL EXTRACTED PAPER:\n" + sample["paper_text"], encoding="utf-8")
    review_csv = work / "rq2_review_300_papers.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (work / "README.txt").write_text(
        "II2202 RQ2 300-PAPER REVIEW\n"
        "IMPORTANT: This is the revised 300 DISTINCT PAPER sample: 272 dev + 28 train.\n"
        "DO NOT use the earlier 300-question/272-paper ZIP as the final sample.\n\n"
        "Review rq2_review_300_papers.csv and fill final_label, final_reason, evidence_location.\n"
        "single-hop: answer needs only one evidence passage.\n"
        "multi-hop: answer truly requires combining separate information within the SAME paper.\n"
        "ambiguous: insufficient/conflicting evidence or cannot reliably classify.\n"
        "Do not infer multi-hop merely from multiple evidence annotations.\n"
        "Check original PDFs where tables are missing. Preserve existing reference answers.\n"
        "Return the completed CSV, and merge by question_id (not row position).\n",
        encoding="utf-8")
    notes = DATA / "rq2_review_notes.md"
    if notes.exists():
        (work / "rq2_review_notes.md").write_bytes(notes.read_bytes())
    zipped = base.ROOT / "RQ2_for_teammate_300_papers.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        for file in work.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(work.parent))
    print("Previously classified:", count["previously_labeled"])
    print("Needs teammate review:", count["needs_review"])
    print("UPDATED RQ2 ZIP:", zipped)
    print("Earlier 300-question/272-paper RQ2 package is not compatible with this final sample.")


def main():
    parser = argparse.ArgumentParser(description="300 distinct QASPER papers: dev+train extension")
    parser.add_argument("command", choices=("prepare", "run", "summarize", "pack-rq2"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize, "pack-rq2": pack_rq2}[args.command]()


if __name__ == "__main__":
    main()