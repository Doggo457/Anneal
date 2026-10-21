"""BFCL Java and JavaScript tool calls: a no-worse check for the loop (21 Oct 2026).

The 150 simple_java (100) and simple_javascript (50) items of the Berkeley Function Calling Leaderboard V4, from the
pinned bfcl-eval 2026.3.23 wheel (code and data Apache-2.0; ShishirPatil/gorilla 6ea5797, 23 Mar 2026). Each item is
one question and one function; the answer is one call to it. Asked in function-calling mode, the way BFCL asks an
OpenAI-compatible server: the question as the only message and the function as a tool, with BFCL's language hint on
every description, every parameter typed as a string and dots in names turned into underscores
(bfcl_eval/utils.py _func_doc_language_specific_pre_processing, model_handler/utils.py convert_to_tool; rebuilt here
and checked equal to BFCL's own output for all 150 items). llama-server parses the call out of the answer (thinking on,
the same server settings as the gate); BFCL's own AST checker judges it against the raw function doc, mapping the
underscores back to dots.

The checker is imported straight from the wheel (zip import), with one module swapped: the wheel's
constants/model_config.py imports every model handler BFCL has, and ast_checker only reads one flag from it
(underscore_to_dot), so a two-line table stands in for it. Nothing from BFCL is installed into the training venv
(bfcl-eval pins numpy 1.26.4 and pulls in sentence-transformers and faiss).

Known checker limits (both sides alike, so they cut the items that can show a change, not the fairness): a Java
String[] answer with quoted elements fails (java 35, 78, 83, 85, 86, 89, 90, 91), and an ArrayList of mixed types
can't pass (java 36, 65).
"""

import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import types
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEEL_NAME = "bfcl_eval-2026.3.23-py3-none-any.whl"
WHEEL_URL = ("https://files.pythonhosted.org/packages/ba/41/ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d"
             "2cb2/" + WHEEL_NAME)
WHEEL_SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
WHEEL_PATH = os.path.join(ROOT, "data", "raw", "bfcl", WHEEL_NAME)
CATEGORIES = (("simple_java", "java"), ("simple_javascript", "javascript"))
HARNESS = "bfcl-fc-1"  # changes when the request or the judging changes (the base is then scored again)
CHECKER_MODEL = "dots_to_underscores"  # the stub table's one entry: tool names had "." -> "_"
TOOL_TAG = "<" "tool_call>"  # in an answer's text: a call llama-server did not parse
_LANG = {"java": "JAVA", "javascript": "JAVASCRIPT"}
_LOCK = threading.Lock()
_CHECKER = {}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_wheel(path=WHEEL_PATH, log=print):
    """The pinned wheel at path (downloaded if missing), sha256-checked."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"downloading {WHEEL_URL}")
        tmp = path + ".part"
        with urllib.request.urlopen(WHEEL_URL, timeout=600) as r, open(tmp, "wb") as fh:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        os.replace(tmp, path)
    sha = file_sha(path)
    if sha != WHEEL_SHA256:
        raise RuntimeError(f"{path}: sha256 {sha}, want {WHEEL_SHA256} (delete it and build again)")
    return path


def _jsonl(zf, name):
    return [json.loads(line) for line in zf.read(name).decode("utf-8").splitlines() if line.strip()]


def hinted(functions, language):
    """BFCL's _func_doc_language_specific_pre_processing (bfcl_eval/utils.py) for Java and JavaScript, on a copy."""
    functions = copy.deepcopy(functions)
    hint = {"java": " Note that the provided function is in Java 8 SDK syntax.",
            "javascript": " Note that the provided function is in JavaScript syntax."}[language]
    lang = {"java": "Java", "javascript": "JavaScript"}[language]
    lists = ("ArrayList", "Array") if language == "java" else ("array",)
    for item in functions:
        item["description"] = item["description"] + hint
        for value in item["parameters"]["properties"].values():
            if value["type"] == "any":
                value["description"] += f" This parameter can be of any type of {lang} object in string representation."
            else:
                value["description"] += f" This is {lang} {value['type']} type parameter in string representation."
            if value["type"] in lists:
                value["description"] += (f" The list elements are of type {value['items']['type']}; they are not in "
                                         "string representation.")
                del value["items"]
            if language == "javascript" and value["type"] == "dict" and "properties" in value:
                value["description"] += (" The dictionary entries have the following schema; they are not in string "
                                         f"representation. {json.dumps(value['properties'])}")
                del value["properties"]
            value["type"] = "string"
    return functions


def to_tools(functions):
    """BFCL's convert_to_tool(functions, GORILLA_TO_OPENAPI, OPENAI_COMPLETIONS) for hinted Java / JavaScript docs,
    where every parameter is already a string: dots in the name become underscores and the parameters an object."""
    out = []
    for item in copy.deepcopy(functions):
        item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        out.append({"type": "function", "function": item})
    return out


def build_rows(wheel=WHEEL_PATH):
    """The 150 items in the wheel's order: id, category, language, messages and tools (the request), function (the
    raw doc the checker reads) and ground_truth."""
    rows = []
    with zipfile.ZipFile(wheel) as zf:
        for category, language in CATEGORIES:
            items = _jsonl(zf, f"bfcl_eval/data/BFCL_v4_{category}.json")
            answers = _jsonl(zf, f"bfcl_eval/data/possible_answer/BFCL_v4_{category}.json")
            truth = {t["id"]: t["ground_truth"] for t in answers}
            for it in items:
                rows.append({"id": it["id"], "category": category, "language": language,
                             "messages": copy.deepcopy(it["question"][0]),
                             "tools": to_tools(hinted(it["function"], language)),
                             "function": it["function"], "ground_truth": truth[it["id"]]})
    return rows


def checker(wheel=WHEEL_PATH):
    """(ast_checker, Language) from the pinned wheel, with the stub model table. Call once before scoring threads
    start."""
    with _LOCK:
        if _CHECKER.get("wheel") == wheel:
            return _CHECKER["fn"], _CHECKER["lang"]
        if not os.path.exists(wheel) or file_sha(wheel) != WHEEL_SHA256:
            raise RuntimeError(f"{wheel} missing or not the pinned bfcl-eval wheel: run scripts/build_guards.py --bfcl")
        stub = types.ModuleType("bfcl_eval.constants.model_config")
        stub.MODEL_CONFIG_MAPPING = {CHECKER_MODEL.replace("_", "/"): types.SimpleNamespace(underscore_to_dot=True)}
        sys.modules["bfcl_eval.constants.model_config"] = stub
        if wheel not in sys.path:
            sys.path.append(wheel)
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval import ast_checker as A
        where = os.path.normcase(os.path.abspath(A.__file__))
        if not where.startswith(os.path.normcase(os.path.abspath(wheel))):
            raise RuntimeError(f"bfcl_eval was imported from {A.__file__}, not from {wheel}")
        _CHECKER.update(wheel=wheel, fn=A.ast_checker, lang=Language)
        return A.ast_checker, Language


def prepare():
    checker()


def request(url, item, seed, max_tokens, sampling, timeout=6 * 3600):
    """One chat completion with the item's tool, thinking on. Returns what evalsets.score keeps, plus the calls."""
    body = {"model": "local", "messages": item["messages"], "tools": item["tools"], "parallel_tool_calls": True,
            "max_tokens": max_tokens, "seed": seed, "chat_template_kwargs": {"enable_thinking": True},
            "stream": False}
    body.update(sampling)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": calls, "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "seconds": round(time.time() - t0, 2),
            "answer": json.dumps(calls, ensure_ascii=False) if calls else (msg.get("content") or "")}


def decode(tool_calls):
    """llama-server's tool_calls -> BFCL's [{name: {arg: value}}], or (None, why)."""
    out = []
    for c in tool_calls:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            return None, "bad_arguments_json"
        if not isinstance(args, dict) or not fn.get("name"):
            return None, "wrong_output_format"
        out.append({fn["name"]: args})
    return out, None


def judge(item, r):
    """(passed, verdict) for one answer: BFCL's AST check of the parsed calls; no call, an unfinished answer or bad
    arguments fail."""
    fin = r.get("finish_reason")
    if fin not in ("tool_calls", "stop"):
        return False, f"finish_{fin}"
    calls = r.get("tool_calls") or []
    if not calls:
        return False, "no_tool_call" + (":unparsed" if TOOL_TAG in (r.get("content") or "") else "")
    decoded, why = decode(calls)
    if why:
        return False, why
    fn, lang = checker()
    with _LOCK:
        res = fn(item["function"], decoded, item["ground_truth"], lang[_LANG[item["language"]]], item["category"],
                 CHECKER_MODEL)
    return bool(res["valid"]), ("ok" if res["valid"] else str(res.get("error_type") or "invalid"))


def probe(rows):
    """After the first answers of a pass: an error when none made a call at all (the server or the chat template is
    not parsing tool calls, so a whole pass would fail), else None."""
    called = [r for r in rows if not str(r.get("verdict", "")).startswith(("no_tool_call", "finish_"))]
    missed = [r for r in rows if str(r.get("verdict", "")).startswith("no_tool_call")]
    if rows and not called and 2 * len(missed) >= len(rows):
        return (f"BFCL: none of the first {len(rows)} answers made a tool call (verdicts "
                f"{sorted({r.get('verdict') for r in rows})}): llama-server is not parsing tool calls here")
    return None


def language(item):
    return item["language"]
