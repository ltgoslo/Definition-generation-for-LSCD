#!/bin/bash

for LANG in english norwegian1 norwegian2; do
    zcat ../generated_definitions/$LANG/greedy/$LANG-corpus1.tsv.gz ../generated_definitions/$LANG/greedy/$LANG-corpus2.tsv.gz | cut -f 1 | sort -u | grep -v 'Word' | grep -v '0' > $LANG/target_words.txt
done

LANG=russian
zcat ../generated_definitions/$LANG/greedy/$LANG-corpus1.tsv.gz ../generated_definitions/$LANG/greedy/$LANG-corpus2.tsv.gz ../generated_definitions/$LANG/greedy/$LANG-corpus3.tsv.gz | cut -f 1 | sort -u | grep -v 'Word' | grep -v '0' > $LANG/target_words.txt
