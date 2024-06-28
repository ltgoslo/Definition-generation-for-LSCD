#!/bin/bash

# module use -a /cluster/shared/nlpl/software/eb/etc/all/
# module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
# module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

for OUTLANG in english norwegian1 norwegian2 russian12 russian13 russian23; do
	mkdir -p $OUTLANG

	if [[ $OUTLANG == russian* ]]
	then
		INLANG=${OUTLANG:0:7}
		A=${OUTLANG:7:1}
		B=${OUTLANG:8:1}
	else
		INLANG=$OUTLANG
		A=1
		B=2
	fi

	echo "$INLANG $A $B => $OUTLANG"

    for RUN in greedy beam diverse_beam_search; do
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
