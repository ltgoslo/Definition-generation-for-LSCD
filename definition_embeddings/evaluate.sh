#! /bin/bash

# module use -a /cluster/shared/nlpl/software/eb/etc/all/
# module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
# module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

declare -A MAPPINGS=( ["english"]="english" ["norwegian1"]="norwegian1" ["norwegian2"]="norwegian2" ["russian12"]="russian1" ["russian13"]="russian3" ["russian23"]="russian2")

for LANG in "${!MAPPINGS[@]}"; do
    for RUN in greedy beam diverse_beam_search; do
        for METHOD in apd prt am gm; do
            echo "** $LANG / ${MAPPINGS[$LANG]} $RUN $METHOD **"
            if [ -f $LANG/$RUN.$METHOD.binary.tsv ]; then
                python ../src/eval.py 1 $LANG/$RUN.$METHOD.binary.tsv ../src/data/"${MAPPINGS[$LANG]}"/truth/binary.txt
            fi
            python ../src/eval.py 2 $LANG/$RUN.$METHOD.csv ../src/data/"${MAPPINGS[$LANG]}"/truth/graded.txt
        done
    done
done
