#!/usr/bin/env bash
pandoc -f markdown+tex_math_dollars+raw_tex+pipe_tables+table_captions+link_attributes \
       -H packages.tex \
       --pdf-engine pdflatex report.md \
       -V geometry:"top=1in,bottom=1in,left=1in,right=1in" \
       -V fontsize=12pt \
       -V colorlinks=true \
       -V linkcolor=blue \
       -V urlcolor=blue \
       -V toccolor=gray \
       -o report.pdf
 
