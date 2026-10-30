"""II2202 real QASPER RAG vs LCP experiment.

Commands:
  python experiment.py prepare
  python experiment.py run --limit 6
  python experiment.py run --limit 150
  python experiment.py summarize

Requires: requests, tiktoken, rank-bm25.
Ollama model: gemma3:4b.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import string
import tarfile
import time
from collections import Counter
from pathlib import Path

import requests
import tiktoken
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
ARCHIVE = DATA / "qasper-train-dev-v0.3.tgz"
SAMPLES = DATA / "qasper_150.jsonl"
LABELS = DATA / "rq2_labels.csv"
SOURCE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
ENCODING = "cl100k_base"  # fixed length-grouping proxy, NOT the Gemma tokenizer
ENCODER = tiktoken.get_encoding(ENCODING)
SEED = 42
NUM_CTX = 32768
NUM_PREDICT = 160
MAX_DOC_PROXY_TOKENS = 16000  # conservative filter for initial 32k context test
CHUNK_WORDS = 240
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_qasper():
    if not ARCHIVE.exists():
        print("Downloading official QASPER v0.3 train/dev archive (~11 MB)...")
        with requests.get(SOURCE_URL, stream=True, timeout=(20, 180)) as res:
            res.raise_for_status()
            with ARCHIVE.open("wb") as out:
                for block in res.iter_content(chunk_size=1024 * 1024):
                    if block:
                        out.write(block)
    try:
        with tarfile.open(ARCHIVE, mode="r:gz") as t:
            member = next(m for m in t.getmembers()
                          if m.isfile() and m.name.endswith("qasper-dev-v0.3.json"))
            with t.extractfile(member) as f:
                return json.load(f)
    except Exception as exc:
        raise RuntimeError("Could not read QASPER archive. Delete data/*.tgz and retry.") from exc


def paper_text(paper):
    parts = [paper.get("title", ""), paper.get("abstract", "")]
    for sec in paper.get("full_text", []):
        parts.append(sec.get("section_name", ""))
        parts.extend(sec.get("paragraphs", []))
    return "\n\n".join(str(x).strip() for x in parts if x and str(x).strip())


def annotated_answer(a):
    if a.get("unanswerable"):
        return None
    yn = a.get("yes_no")
    if yn is True:
        return "yes"
    if yn is False:
        return "no"
    free = (a.get("free_form_answer") or "").strip()
    if free:
        return free
    spans = a.get("extractive_spans") or []
    if spans:
        return " ".join(str(s).strip() for s in spans if str(s).strip())
    return None


def quantile(values, fraction):
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)]


def prepare(args):
    raw = download_qasper()
    papers = []
    total_questions = 0
    for pid, doc in sorted(raw.items()):
        text = paper_text(doc)
        if not text:
            continue
        n_tokens = len(ENCODER.encode(text))
        if n_tokens > args.max_doc_tokens:
            continue
        qas = []
        for qa in doc.get("qas", []):
            answers, evidence = [], []
            for ann in qa.get("answers", []):
                obj = ann.get("answer") or {}
                ans = annotated_answer(obj)
                ev = [str(e).strip() for e in (obj.get("evidence") or []) if str(e).strip()]
                # This text-only experiment excludes figure/table-only evidence.
                text_ev = [e for e in ev if not e.upper().startswith("FLOAT SELECTED")]
                if ans and text_ev:
                    answers.append(ans)
                    evidence.extend(text_ev)
            if answers and evidence:
                qas.append({"question_id": str(qa["question_id"]),
                            "question": qa["question"],
                            "references": list(dict.fromkeys(answers)),
                            "evidence": list(dict.fromkeys(evidence))})
        if qas:
            total_questions += len(qas)
            papers.append({"paper_id": pid, "title": doc.get("title", ""),
                           "text": text, "proxy_tokens": n_tokens,
                           "qas": qas})
    if not papers:
        raise RuntimeError("No eligible papers. Verify source and filtering parameters.")

    # Tertiles are computed on eligible PAPER lengths, not question rows.
    lo = quantile([p["proxy_tokens"] for p in papers], 1 / 3)
    hi = quantile([p["proxy_tokens"] for p in papers], 2 / 3)
    if lo >= hi:
        raise RuntimeError("Length tertile boundaries collapsed; inspect the paper lengths.")
    def group(p):
        return "short" if p["proxy_tokens"] <= lo else (
            "medium" if p["proxy_tokens"] <= hi else "long")

    rng = random.Random(SEED)
    samples = {}
    for grp in ("short", "medium", "long"):
        group_papers = [p for p in papers if group(p) == grp]
        rng.shuffle(group_papers)
        for p in group_papers:
            rng.shuffle(p["qas"])
        chosen = []
        # One question per paper first, then second/third questions if necessary.
        for qi in range(args.per_paper):
            for p in group_papers:
                if len(chosen) >= 50:
                    break
                if qi >= len(p["qas"]):
                    continue
                q = p["qas"][qi]
                chosen.append({"paper_id": p["paper_id"], "title": p["title"],
                               "paper_text": p["text"], "proxy_tokens": p["proxy_tokens"],
                               "length_group": grp, **q})
            if len(chosen) == 50:
                break
        if len(chosen) < 50:
            raise RuntimeError(f"Only {len(chosen)} eligible {grp} questions (need 50). "
                               "Increase --max-doc-tokens or --per-paper, then rerun prepare.")
        samples[grp] = chosen

    # Interleave short / medium / long so the six-question pilot covers all tiers.
    out = [samples[g][i] for i in range(50) for g in ("short", "medium", "long")]
    assert len(out) == 150 and len({r['question_id'] for r in out}) == 150
    write_jsonl(SAMPLES, out)
    with LABELS.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "length_group", "question", "evidence_preview",
                    "hop_label", "reason"])
        for r in out:
            w.writerow([r["question_id"], r["length_group"], r["question"],
                        " | ".join(e[:160] for e in r["evidence"]), "", ""])

    meta = {"dataset": "QASPER v0.3", "split": "dev/validation", "seed": SEED,
            "sample_count": len(out), "distinct_papers": len({r['paper_id'] for r in out}),
            "eligible_papers": len(papers), "eligible_questions": total_questions,
            "length_tokenizer": ENCODING, "length_measurement": "proxy tokens; not Gemma tokens",
            "short_upper": lo, "medium_upper": hi,
            "length_group_counts": dict(Counter(r["length_group"] for r in out)),
            "max_doc_proxy_tokens": args.max_doc_tokens,
            "max_questions_per_paper": args.per_paper,
            "exclusion": "unanswerable/non-text-evidence questions and papers above context proxy cap"}
    (DATA / "sampling_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nSampling OK:", meta["sample_count"], "questions from", meta["distinct_papers"], "papers")
    print("Tertiles (cl100k proxy tokens):", lo, hi)
    print("Saved:", SAMPLES, "and", LABELS)
    print("Open rq2_labels.csv to classify questions as single-hop/multi-hop/ambiguous.")


def chunks(text):
    words = text.split()
    stride = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), stride)]


def retrieval(question, document):
    passages = chunks(document)
    def tok(s):
        return re.findall(r"\b\w+\b", s.lower())
    bm25 = BM25Okapi([tok(p) for p in passages])
    scores = bm25.get_scores(tok(question))
    ranks = sorted(range(len(passages)), key=lambda i: (-scores[i], i))[:TOP_K]
    return "\n\n".join(f"[Passage {i + 1}]\n{passages[i]}" for i in ranks)


def question_prompt(question, context):
    return ("Answer the question using only the supplied paper text. "
            "Give a short factual answer. If the paper does not contain the answer, "
            "write 'Not stated'. Do not add explanation.\n\n"
            f"PAPER TEXT:\n{context}\n\nQUESTION: {question}\nANSWER:")


def ollama(prompt, ctx):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "keep_alive": "15m", "options": {"num_ctx": ctx, "num_predict": NUM_PREDICT,
                                   "temperature": 0, "seed": SEED}},
        timeout=(10, 900))
    response.raise_for_status()
    return response.json()


def norm(text):
    text = str(text).lower()
    text = "".join(c if c not in string.punctuation else " " for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(answer, golds):
    pred = norm(answer)
    best_em, best_f1 = 0, 0.0
    for gold in golds:
        target = norm(gold)
        em = int(pred == target)
        pa, ga = Counter(pred.split()), Counter(target.split())
        common = sum((pa & ga).values())
        if not pa and not ga:
            f1 = 1.0
        elif common == 0:
            f1 = 0.0
        else:
            precision = common / sum(pa.values())
            recall = common / sum(ga.values())
            f1 = 2 * precision * recall / (precision + recall)
        best_em, best_f1 = max(best_em, em), max(best_f1, f1)
    return best_em, round(best_f1, 6)


def run(args):
    if not SAMPLES.exists():
        raise RuntimeError("Run: python experiment.py prepare  FIRST")
    samples = read_jsonl(SAMPLES)
    if args.limit <= 0 or args.limit > len(samples):
        raise ValueError("--limit must be between 1 and 150")
    selected = samples[:args.limit]
    output = DATA / ("pilot_results.jsonl" if args.limit < 150 else "real_results.jsonl")
    existing = read_jsonl(output) if output.exists() else []
    config = {"model": MODEL, "num_ctx": args.ctx, "num_predict": NUM_PREDICT,
              "temperature": 0, "seed": SEED, "top_k": TOP_K,
              "chunk_words": CHUNK_WORDS, "overlap_words": CHUNK_OVERLAP_WORDS}
    if any(row.get("config") != config for row in existing):
        raise RuntimeError("Previous results used different configuration; move/rename results file before rerunning.")
    completed = {(r["question_id"], r["method"]) for r in existing}
    try:
        test = requests.get("http://localhost:11434/api/tags", timeout=5)
        test.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("Ollama is not reachable. Start Ollama on Windows.") from exc

    for j, q in enumerate(selected, 1):
        # Deterministically vary run order to reduce a systematic order bias.
        start_with_rag = int(hashlib.sha256(q["question_id"].encode()).hexdigest(), 16) % 2 == 0
        methods = ["RAG", "LCP"] if start_with_rag else ["LCP", "RAG"]
        for method in methods:
            if (q["question_id"], method) in completed:
                continue
            print(f"[{j}/{len(selected)}] {method}: {q['question_id']}", flush=True)
            started = time.perf_counter()
            context = retrieval(q["question"], q["paper_text"]) if method == "RAG" else q["paper_text"]
            prompt = question_prompt(q["question"], context)
            # Proxy guard. The Ollama actual prompt count is checked after generation too.
            proxy = len(ENCODER.encode(prompt))
            if proxy > int(args.ctx * 0.72):
                raise RuntimeError(f"{q['question_id']} {method}: prompt ~{proxy} proxy tokens "
                                   f"(ctx {args.ctx}), possible truncation. "
                                   "Increase --ctx / reduce eligible document cap and redo prepare.")
            result = ollama(prompt, args.ctx)
            wall = time.perf_counter() - started
            actual = result.get("prompt_eval_count")
            if actual is None or actual > int(args.ctx * 0.9):
                raise RuntimeError(f"{q['question_id']} {method}: actual prompt tokens "
                                   f"{actual} near limit {args.ctx}; reject possible truncation.")
            pred = (result.get("response") or "").strip()
            em, f1 = score(pred, q["references"])
            row = {
                "question_id": q["question_id"], "paper_id": q["paper_id"],
                "length_group": q["length_group"], "document_proxy_tokens": q["proxy_tokens"],
                "method": method, "prediction": pred, "references": q["references"],
                "EM": em, "F1": f1, "latency_sec": round(wall, 4),
                "input_tokens": actual, "output_tokens": result.get("eval_count"),
                "total_tokens": actual + (result.get("eval_count") or 0),
                "ollama_total_sec": round(result.get("total_duration", 0) / 1e9, 4),
                "ollama_load_sec": round(result.get("load_duration", 0) / 1e9, 4),
                "proxy_prompt_tokens": proxy, "config": config,
            }
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            completed.add((q["question_id"], method))
            print(f"   EM={em} F1={f1:.3f} time={wall:.1f}s "
                  f"tokens={row['total_tokens']}", flush=True)
    print("Finished:", output)


def summarize(args):
    path = DATA / "real_results.jsonl"
    if not path.exists():
        path = DATA / "pilot_results.jsonl"
    if not path.exists():
        raise RuntimeError("No results. Run: python experiment.py run --limit 6")
    rows = read_jsonl(path)
    print("\nResults source:", path)
    print("Rows:", len(rows), "distinct questions:", len({r['question_id'] for r in rows}))
    for group in ("short", "medium", "long", "ALL"):
        for method in ("RAG", "LCP"):
            chosen = [r for r in rows if r["method"] == method
                      and (group == "ALL" or r["length_group"] == group)]
            if not chosen:
                continue
            mean = lambda field: sum(x[field] for x in chosen) / len(chosen)
            print(f"{group:6} {method:3} n={len(chosen):3} "
                  f"EM={mean('EM'):.3f} F1={mean('F1'):.3f} "
                  f"latency={mean('latency_sec'):.2f}s tokens={mean('total_tokens'):.0f}")
    # Export flat CSV for teammate to make plots.
    with (DATA / "results_flat.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["question_id", "paper_id", "length_group", "document_proxy_tokens",
                "method", "prediction", "EM", "F1", "latency_sec",
                "input_tokens", "output_tokens", "total_tokens"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("Saved: data/results_flat.csv")
    print("RQ2 classification is manual: fill data/rq2_labels.csv; do NOT infer multi-hop from evidence count.")
    if path.name != "real_results.jsonl" or len(rows) != 300:
        print("PILOT / INCOMPLETE: no report-ready charts generated.")
        return
    pairs = {(r["question_id"], r["method"]) for r in rows}
    if len(pairs) != 300:
        print("INCOMPLETE: duplicate/missing paired runs; charts skipped.")
        return

    import matplotlib.pyplot as plt
    groups = ["short", "medium", "long"]
    def avg_for(group, method, field):
        vals = [r[field] for r in rows if r["length_group"] == group
                and r["method"] == method]
        return sum(vals) / len(vals) if vals else float("nan")

    # Line charts with one point per document-length group.
    def length_chart(field, ylabel, filename):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for method in ("RAG", "LCP"):
            ax.plot(groups, [avg_for(g, method, field) for g in groups],
                    marker="o", label=method)
        ax.set(xlabel="Document-length group", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(DATA / filename, dpi=180)
        plt.close(fig)

    length_chart("F1", "Mean token-level F1", "f1_vs_length_real.png")
    length_chart("EM", "Mean Exact Match", "em_vs_length_real.png")
    length_chart("latency_sec", "Mean end-to-end latency (s)", "latency_vs_length_real.png")
    length_chart("total_tokens", "Mean input + output tokens", "tokens_vs_length_real.png")
    print("Saved 4 REAL QASPER length-group charts in data/ (without CI).")

    labels = {}
    if LABELS.exists():
        with LABELS.open(encoding="utf-8-sig", newline="") as f:
            for line in csv.DictReader(f):
                lab = line.get("hop_label", "").strip().lower()
                if lab in ("single-hop", "multi-hop"):
                    labels[line["question_id"]] = lab
    category_rows = {"single-hop": [], "multi-hop": []}
    for r in rows:
        label = labels.get(r["question_id"])
        if label:
            category_rows[label].append(r)
    if not all(any(r["method"] == "RAG" for r in category_rows[l])
               and any(r["method"] == "LCP" for r in category_rows[l])
               for l in category_rows):
        print("RQ2 chart skipped: fill hop_label as single-hop/multi-hop in rq2_labels.csv.")
        return
    cats = list(category_rows)
    x = list(range(len(cats)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, off in (("RAG", -0.18), ("LCP", 0.18)):
        vals = []
        for label in cats:
            metrics = [r["F1"] for r in category_rows[label] if r["method"] == method]
            vals.append(sum(metrics) / len(metrics))
        ax.bar([v + off for v in x], vals, width=0.35, label=method)
    ax.set_xticks(x, cats)
    ax.set(ylabel="Mean token-level F1", xlabel="Reasoning type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA / "f1_by_hop_real.png", dpi=180)
    plt.close(fig)
    print("Saved REAL RQ2 chart: data/f1_by_hop_real.png")


def main():
    ap = argparse.ArgumentParser(description="QASPER real RAG/LCP experiment (Ollama local)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--max-doc-tokens", type=int, default=MAX_DOC_PROXY_TOKENS)
    p.add_argument("--per-paper", type=int, default=3)
    p = sub.add_parser("run")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--ctx", type=int, default=NUM_CTX)
    sub.add_parser("summarize")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()