#! /bin/bash

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# module purge  # Reset the modules to the system default
# module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
# module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

for LANG in english norwegian1 norwegian2 russian1 russian2 russian3; do
    for RUN in greedy beam divbeam; do
        for METHOD in apd prt am gm; do
            echo "** $LANG $RUN $METHOD **"
            if [ -f $LANG/$RUN.$METHOD.binary.tsv ]; then
                python ../src/eval.py 1 $LANG/$RUN.$METHOD.binary.tsv ../src/data/$LANG/truth/binary.txt
            fi
            python ../src/eval.py 2 $LANG/$RUN.$METHOD.csv ../src/data/$LANG/truth/graded.txt
        done
    done
done
