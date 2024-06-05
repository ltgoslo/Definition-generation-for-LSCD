#! /bin/bash

LANG=${1}
TYPE=${2}
INPUT=${3}  # Path to the input data

echo "Cosine distance:"
python3 eval.py 2 ${INPUT}/${LANG}/${TYPE}/defgen/${LANG}/Cosine_dict.tsv data/${LANG}/truth/graded.txt
echo "JS distance:"
python3 eval.py 2 ${INPUT/}${LANG}/${TYPE}/defgen/${LANG}/JS_dict.tsv data/${LANG}/truth/graded.txt
