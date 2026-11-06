#!/usr/bin/env bash
# Package the paper for arXiv: main.tex, main.bbl, the CoRL style and bibliography style, generated tables and the
# PDF figures it includes. Builds main.pdf first so that main.bbl is current, then test-compiles the package
# in a temporary directory with pdflatex only (no BibTeX), as arXiv does.
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/figures" "$stage/generated"
cp main.tex main.bbl corl_2026.sty corlabbrvnat.bst "$stage/"
cp generated/*.tex "$stage/generated/"
# only the figures that main.tex includes
for f in $(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' main.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do
  cp "figures/$f" "$stage/figures/"
done

# test-compile the staged sources exactly as arXiv would
(cd "$stage" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
rm -f "$stage"/main.{aux,log,out,pdf}

tar -czf arxiv_submission.tar.gz -C "$stage" .
echo "wrote $(pwd)/arxiv_submission.tar.gz"
tar -tzf arxiv_submission.tar.gz | sort
