#! /bin/sh

module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8

# virtual environment with pandas, networkx and Levenshtein modules
source myenv/bin/activate

# joint processing gives much better results than separate processing
MODE=joint

# greedy decoding showed best performance overall
GENMODE=greedy

# selecting hubs by maximum frequency gives better results than selecting by degree
HUB=maxFreq

# additional merging by minimum subsequence length does not provide consistently better results
MIN_SUBSEQ_LEN=-1

# a relative Levenshtein threshold of 0.4 looks optimal across languages
LEVEN_THRESHOLD=0.4

echo "" > results.txt

for LANG in english norwegian1 norwegian2 russian1 russian2 russian3; do
    echo $LANG
    mkdir -p merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"
    if [ "$MODE" = "separate" ] ; then
        python3 graph_merge.py --defgen_path ../generated_definitions/$LANG/$GENMODE --lang $LANG --separate \
            --out_path merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/ \
            --min_subseq_len $MIN_SUBSEQ_LEN --leven_threshold $LEVEN_THRESHOLD --hub_strategy $HUB
    else
        python3 graph_merge.py --defgen_path ../generated_definitions/$LANG/$GENMODE --lang $LANG \
            --out_path merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/ \
            --min_subseq_len $MIN_SUBSEQ_LEN --leven_threshold $LEVEN_THRESHOLD --hub_strategy $HUB
    fi
    python3 ../src/sense_dis.py --data_dir ../src/data --lang $LANG --method defgen --no_path_rewrite \
        --defgen_path merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/ \
        --results_dir merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/ 

    echo $LANG >> results.txt
    python3 ../src/eval.py 2 merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/defgen/$LANG/Cosine_dict.tsv ../src/data/$LANG/truth/graded.txt >> results.txt
    python3 ../src/eval.py 2 merged/$LANG/"$GENMODE"_"$MODE"_"$MIN_SUBSEQ_LEN"_"$LEVEN_THRESHOLD"/defgen/$LANG/JS_dict.tsv ../src/data/$LANG/truth/graded.txt >> results.txt
    echo "" >> results.txt
done

deactivate
