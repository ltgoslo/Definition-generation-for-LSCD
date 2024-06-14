#!/bin/bash

MODEL=${1}  # Definition generation model (e.g. ltg/mt0-definition-en-xl)
TEST=${2}   # Directory containing complete.tsv.gz (input file with examples and target words) or such an input file itself
DATA=${3}   # Where to save the file with generated definitions
PROMPT=${4} # What prompt to use? (see the list in generate_t5.py)

echo ${MODEL}
echo ${TEST}
echo ${PROMPT}
echo ${DATA}

echo "Start generating definitions..."
python3 definition_modeling/code/modeling/generate_t5.py --model ${MODEL} --testdata ${TEST} --save ${DATA} --prompt ${PROMPT} --bsize 16 --maxl 256 --filter 1
echo "Generating definitions finished..."
