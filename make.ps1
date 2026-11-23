#!/usr/bin/env pwsh
# make.ps1 - PowerShell equivalent of the project Makefile.
#
# Usage:
#   ./make.ps1 [target ...]
#
# Targets (default: all):
#   all | pdf   compile the paper and copy a versioned PDF into $OUTDIR
#   tagged      only copy the versioned PDF
#   html        build HTML via latexml / latexmlpost
#   docx        build docx (runs html first, then pandoc)
#   diff        build a diff PDF against the master branch
#   clean       remove all generated intermediate files
#
# Note: if the execution policy blocks scripts, run once:
#   Set-ExecutionPolicy -Scope Process Bypass

param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Targets
)

$ErrorActionPreference = "Stop"

$SOURCE      = "main"
$SOURCE_FILE = "$SOURCE.tex"
$TARGET      = "Paper-DLAN"
$OUTDIR      = "_out"
$DIFF_TARGET = "master"

# DOC_VERSION = last part of the current branch name (text after the final '-')
$branch       = (git rev-parse --abbrev-ref HEAD).Trim()
$DOC_VERSION  = ($branch -split '-')[-1]

function Invoke-Step {
    param([scriptblock]$Step)
    Write-Host "==> $($Step.ToString())" -ForegroundColor Cyan
    & $Step
    if (${LASTEXITCODE} -ne $null -and ${LASTEXITCODE} -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $($Step.ToString())"
    }
}

function Invoke-Html {
    Invoke-Step { latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$OUTDIR" "$SOURCE_FILE" }
    Invoke-Step { latexml --preload=amsmath --preload=html.ltxml --path="$OUTDIR" --dest="$OUTDIR/$TARGET.xml" "$SOURCE_FILE" }
    Push-Location $OUTDIR
    try {
        Invoke-Step { latexmlpost --dest="$TARGET.html" --javascript=LaTeXML-maybeMathjax.js "$TARGET.xml" }
    } finally {
        Pop-Location
    }
}

if ($Targets.Count -eq 0) {
    $Targets = @("all")
}

foreach ($t in $Targets) {
    Write-Host "==> Target: $t" -ForegroundColor Yellow
    switch ($t) {
        { $_ -in @("all", "pdf") } {
            Invoke-Step { latexmk -pdf -outdir=$OUTDIR $SOURCE }
            Invoke-Step { Copy-Item "$OUTDIR/$SOURCE.pdf" "$OUTDIR/${TARGET}_${DOC_VERSION}.pdf" -Force }
        }
        "anonymous" {
            Invoke-Step { latexmk -pdf -jobname="${SOURCE}_anonymous" -pretex='\newif\ifanonymous\anonymoustrue' -usepretex -outdir="${OUTDIR}" "${SOURCE}" }
        }
        "tagged" {
            Invoke-Step { Copy-Item "$OUTDIR/$SOURCE.pdf" "$OUTDIR/${TARGET}_${DOC_VERSION}.pdf" -Force }
        }
        "html" {
            Invoke-Html
        }
        "docx" {
            Invoke-Html
            Push-Location $OUTDIR
            try {
                Invoke-Step { pandoc -o "${TARGET}_${DOC_VERSION}.docx" --reference-doc=template.docx --mathml "$TARGET.html" }
            } finally {
                Pop-Location
            }
        }
        "diff" {
            Invoke-Step { latexdiff-vc --force --exclude-textcmd=section,subsection,subsubsection --append-textcmd=significancestatement,makecell --add-to-config ld.cfg --math-markup=0 -r $DIFF_TARGET $SOURCE_FILE }
            Invoke-Step { Move-Item "${SOURCE}-diff${DIFF_TARGET}.tex" "${TARGET}_diff${DIFF_TARGET}_${DOC_VERSION}.tex" -Force }
            Invoke-Step { latexmk -pdf -outdir=$OUTDIR "${TARGET}_diff${DIFF_TARGET}_${DOC_VERSION}.tex" }
        }
        "clean" {
            Invoke-Step { latexmk -C -outdir=$OUTDIR }
            foreach ($ext in @("aux", "log", "bbl", "blg", "xml", "html")) {
                Remove-Item "$OUTDIR/*.$ext" -ErrorAction SilentlyContinue
            }
        }
        default {
            throw "Unknown target: $t"
        }
    }
}
