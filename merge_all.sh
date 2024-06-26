#!/bin/bash

for LANG in russian2 russian3 norwegian1 norwegian2 russian1; do
  for GEN in greedy beam divbeam; do
    for MERGE in full_fledged minimalist; do
        ./merge_pipeline.sh ${LANG} ${GEN} ${MERGE} 10
    done
  done
done
for GEN in greedy beam divbeam; do
  for MERGE in full_fledged minimalist; do
    ./merge_pipeline.sh english ${GEN} ${MERGE} 50
  done
done