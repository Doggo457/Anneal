# Outline build source

Builds `Anneal_Dissertation_Outline.docx` from the university Word template.

- `content.py`: all outline text, tables, listing and references (edit this). Markup: `**bold**`, `*italic*`, `[@key]` citation, `{{text}}` yellow gap, `` `code` ``.
- `build.py`: fills the university's outline template from `content.py` (python-docx; IEEE refs numbered by first citation).
- `tpl/template.docx`: the university's Word template for the outline. It is the university's file, so git ignores it: save a copy there before building.
- `fig.py` + `shot.js`: draw Figure 1 (`fig1.png`); only needed if the figure changes (needs Node + Playwright).
- `wcdocx.py`: word count per chapter, Introduction heading to before References (hard maximum 4,000).

Rebuild (run from this folder; needs `pip install python-docx`):

    python3 build.py . tpl/template.docx ../Anneal_Dissertation_Outline.docx
    python3 wcdocx.py ../Anneal_Dissertation_Outline.docx

Redraw the figure first, if changed: `python3 fig.py && node shot.js "$PWD/fig1.html" "$PWD/fig1.png"`
