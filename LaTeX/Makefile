.PHONY: all clean pdf view

LATEX = pdflatex
BIBTEX = bibtex
MAIN = main

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf &

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz
	rm -f $(MAIN).pdf

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all     - Build the PDF (default)"
	@echo "  pdf     - Build PDF document"
	@echo "  view    - Open the PDF viewer"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"
