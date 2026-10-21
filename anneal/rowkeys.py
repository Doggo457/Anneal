"""Row identity shared by the loop (scripts/loop.py) and the reworded-copy check (scripts/paraphrase_check.py).

row_key decides what counts as "the same data" when the loop hands out rows, so both scripts must agree on it.
Standard library only: the check runs beside training and must not import the loop's GPU-side modules.
"""
import hashlib
import json


def row_key(row):
    """What makes a row "the same data": the question for single-turn rows, the first user message + the trained
    turn for multi-turn rows (each step of an agent session is its own row)."""
    msgs = row.get("messages") or []
    users = [m for m in msgs if m.get("role") == "user"]
    if not users:
        return None
    if len(users) == 1 and sum(m.get("role") == "assistant" for m in msgs) == 1:
        text = users[0].get("content") or ""
    else:
        last = msgs[-1]
        text = (users[0].get("content") or "") + "\n#\n" + (last.get("content") or "") + json.dumps(last.get("tool_calls") or [])
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def question_text(row):
    """The task a row asks for: its first user message as plain text ("" if it has none). Content given as a list of
    parts keeps only the text parts."""
    for m in row.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
        return "" if c is None else str(c)
    return ""


def message_reasoning(msg):
    """The reasoning of one assistant message, stripped ("" if none): a reasoning_content / reasoning / thinking field,
    or a "<think>TRACE</think>" block that opens the content (the pools' form). The block test is the one
    scripts/train_longrun.py split_trace applies, which trains a row in thinking mode only when TRACE is non-empty."""
    for k in ("reasoning_content", "reasoning", "thinking"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    c = msg.get("content")
    s = c.lstrip() if isinstance(c, str) else ""
    if not s.startswith("<think>") or "</think>" not in s:
        return ""
    return s[len("<think>"):s.index("</think>")].strip()


def reasoning_text(row):
    """The teacher's reasoning for the turn a row trains (its last assistant message); "" if it has none. Since
    21 Oct 2026 a row without reasoning is never trained on."""
    for m in reversed(row.get("messages") or []):
        if m.get("role") == "assistant":
            return message_reasoning(m)
    return ""


LANG_ALIASES = {
    "python": "python", "py": "python", "python3": "python", "py3": "python", "python2": "python", "py2": "python",
    "ipython": "python", "pycon": "python",
    "cpp": "cpp", "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "c": "c", "h": "c", "java": "java",
    "javascript": "javascript", "js": "javascript", "jsx": "javascript", "node": "javascript", "mjs": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript", "go": "go", "golang": "go", "rust": "rust",
    "rs": "rust", "csharp": "csharp", "cs": "csharp", "c#": "csharp", "kotlin": "kotlin", "kt": "kotlin",
}


def answer_text(content):
    """The answer part of an assistant message: what follows a leading <think>...</think> block (all of it if none)."""
    s = (content or "").lstrip()
    if s.startswith("<think>") and "</think>" in s:
        return s.split("</think>", 1)[1]
    return content or ""


def fence_languages(text):
    """The languages of the fenced code blocks in text, in order, by their opening fence's tag (lowercase, aliases
    folded: py/python3 -> python, c++/cpp -> cpp, js -> javascript, ...; "" for an untagged block). Opening and
    closing fences alternate, so a closing fence is never read as a language."""
    out, open_fence = [], None
    for line in (text or "").splitlines():
        s = line.strip()
        if not (s.startswith("```") or s.startswith("~~~")):
            continue
        mark = s[:3]
        if open_fence is None:
            tag = s.lstrip(mark[0]).strip().split()[0].strip("{}.").lower() if s.lstrip(mark[0]).strip() else ""
            out.append(LANG_ALIASES.get(tag, tag))
            open_fence = mark
        elif mark == open_fence and not s.lstrip(mark[0]).strip():
            open_fence = None
    return out
