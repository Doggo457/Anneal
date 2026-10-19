W=810; xs=[4,212,420,628]; w=176
def box(x,y,h,title,subs,key=False):
    cx=x+w/2; n=len(subs)
    block=20+n*17          # title + lines height
    ty=y+(h-block)/2+12
    out=[f'<rect class="{"key" if key else "box"}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>',
         f'<text class="t" x="{cx}" y="{ty:.0f}">{title}</text>']
    for i,s in enumerate(subs): out.append(f'<text class="s" x="{cx}" y="{ty+21+i*17:.0f}">{s}</text>')
    return '\n'.join(out)
A=[("Training data",["public sets with traces","(round 0 adds own rows)"],False),
   ("Data pipeline",["trace check, deduplicate,","decontaminate, 8,192 tokens"],False),
   ("QLoRA training",["frozen 4-bit base + LoRA,","resumable loop rounds"],True),
   ("Gated adapters",["kept only if the gate","improves; GGUF export"],False)]
B=[("Q4_K_M base model",["Qwen3.8-27B GGUF,","shared by B and A"],False),
   ("Inference (262k)",["llama-server, thinking on","B: base only","A: base + adapter"],False),
   ("Evaluation harness",["HumanEval+, MBPP+,","LiveCodeBench, own tasks,","RULER, LongCodeBench,","non-coding check"],False),
   ("Paired comparison",["same problems, 3 seeds:","paired bootstrap 95% CI,","permutation test"],True)]
ya,ha=40,92; yb,hb=254,112
P=[f'<rect class="lane" x="0" y="4" width="{W}" height="146" rx="8"/>',
   f'<rect class="lane" x="0" y="218" width="{W}" height="{hb+46}" rx="8"/>',
   '<text class="lanelbl" x="10" y="27">Training<tspan class="lanesub" dx="8">AMD GPU (32 GB), PyTorch + ROCm</tspan></text>',
   '<text class="lanelbl" x="10" y="241">Evaluation<tspan class="lanesub" dx="8">same Q4_K_M base for B and A</tspan></text>']
for (t,s,k),x in zip(A,xs): P.append(box(x,ya,ha,t,s,k))
for (t,s,k),x in zip(B,xs): P.append(box(x,yb,hb,t,s,k))
for i in range(3):
    P.append(f'<path class="arr" d="M{xs[i]+w},{ya+ha/2} H{xs[i+1]-2}"/>')
    P.append(f'<path class="arr" d="M{xs[i]+w},{yb+hb/2} H{xs[i+1]-2}"/>')
ax=xs[3]+w/2; bx=xs[1]+w/2; tx=xs[2]+w/2
P.append(f'<path class="arr" d="M{ax},{ya} V30 H{tx} V{ya-2}"/>')
P.append(f'<text class="lbl" x="{(ax+tx)/2}" y="25" text-anchor="middle">next round from the best kept adapter</text>')
P.append(f'<path class="arr" d="M{ax},{ya+ha} V184 H{bx} V{yb-2}"/>')
P.append(f'<text class="lbl" x="{(ax+bx)/2}" y="178" text-anchor="middle">adapter loaded at runtime</text>')
H=218+hb+46+4
css="""body{margin:0;background:#fff} svg{font-family:Carlito,Calibri,sans-serif;display:block}
.lane{fill:#F3F5F8} .lanelbl{font-size:16px;font-weight:700;fill:#1F2937} .lanesub{font-size:14px;font-weight:400;fill:#4B5563}
.box{fill:#fff;stroke:#374151;stroke-width:1.3} .key{fill:#EAF2FB;stroke:#1F5F9E;stroke-width:1.7}
.t{font-size:16px;font-weight:700;fill:#111827;text-anchor:middle} .s{font-size:13.5px;fill:#374151;text-anchor:middle}
.arr{stroke:#374151;stroke-width:1.5;fill:none;marker-end:url(#a)} .lbl{font-size:13.5px;fill:#1F5F9E;font-style:italic}"""
open('fig1.html','w').write(f'''<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<svg id="fig" width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#374151"/></marker></defs>
{chr(10).join(P)}
</svg></body></html>''')
