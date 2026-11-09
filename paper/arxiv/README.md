# arXiv submission (v1)

The files submitted to arXiv for *Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy's Misreading*.

| File | What it is |
|---|---|
| `arxiv_submission_v1.tar.gz` | LaTeX source as uploaded: `main.tex`, `main.bbl`, style files, figures and generated tables. Built by `../arxiv.sh` from `../main.tex`. |
| `abstract.txt` | The abstract as plain text for arXiv's abstract field. |

arXiv compiles the tarball itself with pdflatex. Three passes give 24 pages, 7 figures, no undefined references. arXiv does not accept a PDF made from LaTeX, so upload the tarball.

To rebuild from the current source: `cd paper && ./arxiv.sh`, then copy `arxiv_submission.tar.gz` here.

## Submission fields

| Field | Value |
|---|---|
| License | CC BY 4.0 |
| Primary category | cs.RO (Robotics) |
| Cross-lists | cs.LG, cs.AI |
| Title | Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy's Misreading |
| Authors | Ali-Adeeb Abbas, Esmaeil Seraj |
| Comments | 24 pages, 7 figures. Project page with videos: https://ali-adeeb.com/weightsmith/ ; code and benchmark: https://github.com/adeeb10abbas/weightsmith |
| Abstract | contents of `abstract.txt` |

Once the paper is announced, add its arXiv ID to the project page, `README.md`, `CITATION.cff` and the BibTeX entries.
