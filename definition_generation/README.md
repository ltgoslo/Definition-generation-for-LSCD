# Generating definitions for usage examples

## Input data
To generate definitions which will be used for semantic change detection, you will first need a corpus of examples (target word usages).
It should be a tab-separated plain text file where the first column contains target words, and the second column contains their usages.
For example:

```
bank <TAB> She sat on the river bank and cried.
```

If you want to evaluate the generated definitions against the gold ones, the input file should contain the gold definitions.
Then it should have at least the following named columns:
- `Targets`
- `Definition`
- `Context`

These columns will be preserved in the output file, and the generated definitions will be added as a separate column.

## Generation

The generation code is maintained in a separate repository.
Clone it if you haven't done that already:

`git clone git@github.com:ltgoslo/definition_modeling.git`

Then you should use the `code/modeling/generate_t5.py` script to generate the actual definitions.
Its 4 most important arguments are:
1. `--model` controls what definition generation model to use (for example, [this one](https://huggingface.co/ltg/mt0-definition-en-xl) for English)
2. `--testdata` is the path to the input data file
3. `--prompt` controls what instruction prompt to use for generation. You can find the prompt ids [here](https://github.com/ltgoslo/definition_modeling/blob/main/code/modeling/generate_t5.py#L234); the id for "What is the definition of <TRG>?" added after the example is `8`.
4. `--save` tells the script where to save the resulting definitions.

For example:

`python3 definition_modeling/code/modeling/generate_t5.py --model ltg/mt0-definition-en-xl --testdata english_corpus.tsv.gz --save english_definitions.tsv.gz --prompt 8 --bsize 16 --maxl 256 --filter 1`

You can also specify other hyperparameters like batch size, maximum definition length, filtering of target words, generation strategies, etc, check `python3 generate_t5.py -h`

## Evaluation

For evaluation, the `code/evaluation/evaluate_simple.py` from the same repository should be used.

You specify the path to the file with the generated definitions, the path to the output file with the evaluation scores, and the language of your data (used in computing BERTScore).

For example:
`python3 definition_modeling/code/evaluation/evaluate_simple.py --data_path english_definitions.tsv.gz --output scores_english.txt -lang en`

_NB: if your input data does not come from WordNet or Oxford dictionary, and thus does not contain sense identifiers, you might want to set `--multiple_definitions_same_sense_id=max` as an argument to the evaluation script.
Otherwise (with the default setting of `mean`), every definition will be evaluated against all gold definitions of its target word, even those belonging to different senses._

## Wrapper script
The [`definition_generation_pipeline.sh`](definition_generation_pipeline.sh) bash script is an easy to use wrapper allowing you to play with models, inputs and prompts, and immediately evaluate the output.
