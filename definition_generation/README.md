# Generating definitions for usage examples

## Input data
To generate definitions which will be used for semantic change detection, you will first need a corpus of examples (target word usages).
It should be a tab-separated plain text file where the first column contains target words, and the second column contains their usages.
For example:

```
bank <TAB> She sat on the river bank and cried.
```

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


The [`definition_generation_pipeline.sh`](definition_generation_pipeline.sh) bash script is an easy to use wrapper allowing you to play with models, inputs and prompts.
