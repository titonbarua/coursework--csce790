#!/usr/bin/env bash
pandoc -f markdown+tex_math_dollars+raw_tex+pipe_tables+table_captions \
       -H packages.tex \
       -V geometry:"top=2cm, bottom=2cm, left=1cm, right=1cm" \
       --pdf-engine pdflatex report.md \
       -V colorlinks=true \
       -V linkcolor=blue \
       -V urlcolor=blue \
       -V toccolor=gray \
       -o report.pdf
 
