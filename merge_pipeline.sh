#!/bin/bash

LANG=${1}  # Data language (english, norwegian1, norwegian2, etc)
GEN=${2}  # Generation strategy (greedy, beam, divbeam)
MERGE=${3} # Merge strategy (full_fledged or minimalist)
THRESHOLD=${4}  # Levenshtein threshold (10 or 50...)

echo ${LANG}
echo ${GEN}
echo ${MERGE}
echo ${THRESHOLD}

cd src/
echo "Merging definitions from ../generated_definitions/${LANG}/${GEN}/..."
if python3 merge_definitions_string.py --data_path ../generated_definitions/${LANG}/${GEN}/ --out ../generated_definitions/merged/${LANG}/${MERGE}/${GEN}${THRESHOLD}/ --lang ${LANG} --thresh ${THRESHOLD} --strategy ${MERGE} ; then
    echo "Definitions merged and saved to ../generated_definitions/merged/${LANG}/${MERGE}/${GEN}${THRESHOLD}/"
else
 echo "Definitions merging failed"
 exit 1
fi

echo "Evaluating..."
if python3 sense_dis.py --data_dir data/ --defgen_path ../generated_definitions/merged/${LANG}/${MERGE}/${GEN}${THRESHOLD}/ --results_dir predictions/merge_results/${LANG}/${GEN}${THRESHOLD}-${MERGE}/ --method defgen --lang ${LANG} ; then
 echo "Evaluation finished"
 ./eval.sh ${LANG} ${GEN}${THRESHOLD}-${MERGE}
else
 echo "Evaluation failed"
fi
