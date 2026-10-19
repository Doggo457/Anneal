"""The first run's second OpenCodeReasoning draw (ocr_first_b, 500 rows), added on 19 Oct 2026 when the run's
self-generated rows were cut from 1,000 to 500. Run from the training root:
  python scripts/first_run_ocr_b.py gate   -> data/eval/ocr_first_plus_gate.jsonl: decontam_extra.jsonl (the loop's
                                              held-out problems) + every ocr_first prompt, for --decontaminate-gate
  python scripts/first_run_ocr_b.py check  -> overlap of data/processed/ocr_first_b with ocr_first and with the
                                              held-out problems (all should be 0)
Between the two, prepare_dataset.py draws the rows with ocr_first's settings, --seed 43 and that gate file.
"""
import json, os, sys
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from anneal.evalsets import OverlapIndex

EXTRA = os.path.join(ROOT, "data", "eval", "decontam_extra.jsonl")
GATE = os.path.join(ROOT, "data", "eval", "ocr_first_plus_gate.jsonl")
A = os.path.join(ROOT, "data", "processed", "ocr_first")
B = os.path.join(ROOT, "data", "processed", "ocr_first_b")


def rows(folder):
    out = []
    for name in ("train", "val"):
        with open(os.path.join(folder, name + ".jsonl"), encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def user(r):
    return r["messages"][-2]["content"]


if sys.argv[1] == "gate":
    with open(EXTRA, encoding="utf-8") as fh:
        extra = [l for l in fh if l.strip()]
    a = rows(A)
    with open(GATE, "w", encoding="utf-8") as fh:
        for l in extra:
            fh.write(l if l.endswith("\n") else l + "\n")
        for i, r in enumerate(a):
            fh.write(json.dumps({"id": f"ocr_first/{i}", "prompt": user(r)}, ensure_ascii=False) + "\n")
    print(f"gate file: {len(extra)} held-out items + {len(a)} ocr_first prompts -> {GATE}")
elif sys.argv[1] == "check":
    a, b = rows(A), rows(B)
    a_q = {user(r).strip().lower() for r in a}
    exact = sum(user(r).strip().lower() in a_q for r in b)
    idx_a = OverlapIndex([(str(i), user(r)) for i, r in enumerate(a)])
    idx_x = OverlapIndex.from_jsonl(EXTRA)
    near_a = sum(bool(idx_a.hits(user(r) + "\n" + r["messages"][-1]["content"], jaccard_text=user(r))) for r in b)
    held = sum(bool(idx_x.hits(user(r) + "\n" + r["messages"][-1]["content"], jaccard_text=user(r))) for r in b)
    b_q = [user(r).strip().lower() for r in b]
    print(json.dumps({"ocr_first_rows": len(a), "ocr_first_b_rows": len(b), "exact_prompt_overlap": exact,
                      "near_overlap_with_ocr_first": near_a, "overlap_with_held_out": held,
                      "repeated_prompts_inside_b": len(b_q) - len(set(b_q))}))
else:
    sys.exit("usage: first_run_ocr_b.py gate|check")
