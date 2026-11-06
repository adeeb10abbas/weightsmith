# Build the paper. `make` compiles main.pdf; `make figures` regenerates figures and generated tables from repo data;
# `make arxiv` builds arxiv_submission.tar.gz; `make clean` removes LaTeX intermediates.
.PHONY: all pdf figures arxiv clean

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	python3 scripts/make_figures.py

arxiv: pdf
	./arxiv.sh

clean:
	latexmk -c main.tex
	rm -f main.run.xml main.bcf
