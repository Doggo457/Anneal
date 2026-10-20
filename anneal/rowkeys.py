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
