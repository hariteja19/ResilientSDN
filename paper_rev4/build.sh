#!/bin/sh
# Build main_rev4.pdf (WSL/Linux with TeX Live) and list errors and undefined references.
cd "$(dirname "$0")"
nice -n 19 latexmk -pdf -interaction=nonstopmode -halt-on-error main_rev4.tex > build.out 2>&1
echo "rc=$?"
grep -E '^!|undefined|Undefined' main_rev4.log | sort -u | head -30
