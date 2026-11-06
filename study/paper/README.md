# Current manuscript and Overleaf sources

This is the only current manuscript directory in the handoff. `paper.pdf` is the checked six-body-page draft plus one reference page. Edit `main.tex`, `sections/`, and `references.bib`; all prebuilt figures are under `figures/`.

Upload the contents of this directory, or the release asset `overleaf.zip`, to a new Overleaf project. Select **XeLaTeX** and `main.tex`. No Python, network fetch, model, GPU, or shell escape is required. The included bibliography output is a fallback; run BibTeX when changing references. The portable Tectonic/XeTeX build was checked locally; Overleaf's server was not used.

The completed Astra evidence is frozen. Add teammate experiments as a separately identified study with model/policy/task, source and checkpoint identities, diagnostic validity, budgets, baseline controls, all proposed edits, held-out results and costs. Motor-versus-perception and policy-family advantages are currently hypotheses. Never reinterpret a pending result as measured or pool incompatible runtimes.

Source measurements are in `../results/`; the narrative and continuation guide start at `../README.md`. Older PDF/LaTeX versions remain in the owner's local archive and are intentionally absent here.
