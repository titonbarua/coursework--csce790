#!/usr/bin/env bash
pandoc -f markdown+tex_math_dollars+raw_tex \
       -H packages.tex \
       -V geometry:"top=2cm, bottom=2cm, left=1cm, right=1cm" \
       --pdf-engine pdflatex report.md \
       -o report.pdf
