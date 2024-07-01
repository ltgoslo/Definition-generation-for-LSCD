#!/bin/bash

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# module purge  # Reset the modules to the system default

# module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
# module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8
# module load nlpl-accelerate/0.27.2-foss-2022b-Python-3.10.8
# module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8

for LANG in english norwegian1 norwegian2 russian; do
    mkdir -p $LANG
    if [ $LANG = "russian" ]; then
        IDS="1 2 3"
    else
        IDS="1 2"
    fi

    for RUN in greedy beam divbeam; do
        for ID in $IDS; do
            echo "$LANG $RUN $ID"
            python embed_definitions.py -m sentence-transformers/all-distilroberta-v1 -i ../generated_definitions/$LANG/$RUN/$LANG-corpus"$ID".tsv* -b 16 -o $LANG/embeddings_perword.$RUN.$ID.npz
        done
    done
done
