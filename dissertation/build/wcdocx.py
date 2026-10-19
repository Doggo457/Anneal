# Word-style count of a built outline: everything between the first Heading 1 and the References heading,
# including headings, table cells, captions and code listings. Cover page and reference list excluded.
import sys, re
from docx import Document
from docx.oxml.ns import qn
d = Document(sys.argv[1]); body = d.element.body
def ptext(p): return "".join(t.text or "" for t in p.iter(qn("w:t")))
def style(p):
    s = p.find(qn("w:pPr")); s = s.find(qn("w:pStyle")) if s is not None else None
    return s.get(qn("w:val")) if s is not None else ""
counts = {}; cur = None; tbl = {}
for el in body:
    paras = [el] if el.tag == qn("w:p") else (list(el.iter(qn("w:p"))) if el.tag == qn("w:tbl") else [])
    for p in paras:
        t = ptext(p); st = style(p)
        if el.tag == qn("w:p") and t.strip() == "References" and not st.startswith("Body"):
            cur = None; break
        if el.tag == qn("w:p") and st.startswith("Heading1"):
            cur = t.strip(); counts[cur] = 0; tbl[cur] = 0
        if cur is None: continue
        n = len(re.findall(r"\S+", t))
        counts[cur] += n
        if el.tag == qn("w:tbl"): tbl[cur] += n
tot = 0
for k, v in counts.items(): print(f"{k:28s} {v:5d}   (of which tables {tbl[k]})"); tot += v
print(f"{'TOTAL':28s} {tot:5d}")
