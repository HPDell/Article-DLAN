SOURCE = main
SOURCE_FILE = $(SOURCE).tex
TARGET = Paper-DLAN
OUTDIR = _out
DIFF_TARGET = master
DOC_VERSION = `git rev-parse --abbrev-ref HEAD | sed -n 's/^.*-//p'`

.PHONY: all pdf html diff clean

all: pdf

pdf:
	latexmk -pdf -outdir=$(OUTDIR) $(SOURCE)
	cp "$(OUTDIR)/$(SOURCE).pdf" "$(OUTDIR)/$(TARGET)_$(DOC_VERSION).pdf"

anonymous:
	latexmk -pdf -jobname=$(TARGET)_$(DOC_VERSION)_anonymous -pretex='\newif\ifanonymous\anonymoustrue' -usepretex -outdir=$(OUTDIR) $(SOURCE)

tagged:
	cp "$(OUTDIR)/$(SOURCE).pdf" "$(OUTDIR)/$(TARGET)_$(DOC_VERSION).pdf"

html:
	latexml --preload=amsmath --dest="$(OUTDIR)/$(TARGET).xml" "$(SOURCE).tex"
	cd "$(OUTDIR)" && latexmlpost --dest="$(TARGET).html" --css=style.css --javascript=LaTeXML-maybeMathjax.js "$(TARGET).xml"

docx: html
	cd "$(OUTDIR)" && pandoc -o "$(TARGET)_$(DOC_VERSION).docx" --reference-doc=template.docx --mathml "$(TARGET).html" 

diff:
	latexdiff-vc --force --exclude-textcmd="section,subsection,subsubsection" --append-textcmd="significancestatement,makecell" --add-to-config ld.cfg --math-markup=0 -r $(DIFF_TARGET) $(SOURCE_FILE)
	mv "$(SOURCE)-diff$(DIFF_TARGET).tex" "$(TARGET)_diff$(DIFF_TARGET)_$(DOC_VERSION).tex"
	latexmk -pdf -outdir=$(OUTDIR) "$(TARGET)_diff$(DIFF_TARGET)_$(DOC_VERSION).tex"

clean:
	latexmk -C -outdir=$(OUTDIR)
	rm -f $(OUTDIR)/*.aux $(OUTDIR)/*.log $(OUTDIR)/*.bbl $(OUTDIR)/*.blg $(OUTDIR)/*.xml $(OUTDIR)/*.html
