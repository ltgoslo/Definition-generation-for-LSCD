#!/bin/bash

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# module purge  # Reset the modules to the system default

# module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
# module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

for OUTLANG in english norwegian1 norwegian2 russian1 russian2 russian3; do
	mkdir -p $OUTLANG

	if [[ $OUTLANG == russian1 ]]; then
		INLANG=russian
		A=1
		B=2
	elif [[ $OUTLANG == russian2 ]]; then
		INLANG=russian
		A=2
		B=3
	elif [ $OUTLANG == russian3 ]]; then
		INLANG=russian
		A=1
		B=3
	else
		INLANG=$OUTLANG
		A=1
		B=2
	fi

	echo "$INLANG $A $B => $OUTLANG"

    for RUN in greedy beam divbeam; do
		python new_scores/average_pairwise_distance.py \
		-i0 ../embed_definitions/$INLANG/embeddings_perword.$RUN.$A.npz \
		-i1 ../embed_definitions/$INLANG/embeddings_perword.$RUN.$B.npz \
		-t ../embed_definitions/$INLANG/target_words.txt \
		-o $OUTLANG/$RUN.apd.csv -f

		python new_scores/cosine_distance.py \
		-i0 ../embed_definitions/$INLANG/embeddings_perword.$RUN.$A.npz \
		-i1 ../embed_definitions/$INLANG/embeddings_perword.$RUN.$B.npz \
		-t ../embed_definitions/$INLANG/target_words.txt \
		-o $OUTLANG/$RUN.prt.csv -f

		python new_scores/combine_scores.py \
		-i0 $OUTLANG/$RUN.apd.csv -i1 $OUTLANG/$RUN.prt.csv -m arithmetic \
		-o $OUTLANG/$RUN.am.csv

		python new_scores/combine_scores.py \
		-i0 $OUTLANG/$RUN.apd.csv -i1 $OUTLANG/$RUN.prt.csv -m geometric \
		-o $OUTLANG/$RUN.gm.csv
	done
done
