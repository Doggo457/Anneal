"""Judge stored gate answers again on another gate file, with today's judge (CPU only; no model needed).

  python scripts/rejudge_gate.py --rows results/loop/rounds/r000/gate.seed1.jsonl --seed 1
                                 --gate data/eval/gate200/gate.jsonl --out results/gate200/base-rejudged.jsonl

Every answer row keeps its whole answer (anneal/evalsets.score), so a round's programs can be built and run again on
the fixed 200's full test sets without asking the model anything. An answer that didn't finish (finish_reason not
"stop") or is empty fails, as when it was scored. Rows of items missing from --rows are listed and left out. The out
rows are the old rows with the new "pass" and "verdict", plus "old_pass" and "old_verdict". Resumable: items already
in --out are skipped. Prints old and new pass counts on the items judged and every item whose verdict changed.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anneal import evalsets as E  # noqa: E402


def rejudge(items, rows, seed, out_path, judge, parallel=2, log=print):
    old = {r["id"]: r for r in rows if r["seed"] == seed}
    missing = [it["id"] for it in items if it["id"] not in old]
    done = {r["id"] for r in E.read_jsonl(out_path)} if os.path.exists(out_path) else set()
    todo = [it for it in items if it["id"] in old and it["id"] not in done]
    log(f"rejudge: {len(items)} items, {len(missing)} without an answer row, {len(done)} done, {len(todo)} to do")

    def one(it):
        r = old[it["id"]]
        if r.get("finish_reason") == "stop" and r.get("answer"):
            ok, verdict = judge(it, r["answer"])
        else:
            ok, verdict = False, r.get("verdict")
        return dict(r, old_pass=r["pass"], old_verdict=r.get("verdict"), **{"pass": bool(ok), "verdict": verdict})

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with ThreadPoolExecutor(max(1, parallel)) as pool, open(out_path, "a", encoding="utf-8") as fh:
        for row in pool.map(one, todo):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    new = [r for r in E.read_jsonl(out_path) if r["seed"] == seed]
    ids = {it["id"] for it in items}
    new = [r for r in new if r["id"] in ids]
    changed = [r for r in new if r["pass"] != r["old_pass"]]
    summary = {"items": len(items), "judged": len(new), "missing": missing,
               "old_pass": sum(r["old_pass"] for r in new), "new_pass": sum(r["pass"] for r in new),
               "fail_to_pass": sorted(r["id"] for r in changed if r["pass"]),
               "pass_to_fail": sorted(r["id"] for r in changed if not r["pass"])}
    log(f"rejudge: {summary['judged']} judged; old judge {summary['old_pass']} passes, new judge "
        f"{summary['new_pass']}; {len(summary['fail_to_pass'])} fail -> pass, {len(summary['pass_to_fail'])} "
        f"pass -> fail")
    for r in sorted(changed, key=lambda r: r["id"]):
        log(f"  {r['id']}: {r['old_verdict']} -> {r['verdict']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=2, help="programs judged at once (each JVM may take 2 GB)")
    args = ap.parse_args()
    summary = rejudge(E.read_jsonl(args.gate), E.read_jsonl(args.rows), args.seed, args.out, E.polyglot_check(),
                      args.parallel)
    with open(os.path.splitext(args.out)[0] + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
