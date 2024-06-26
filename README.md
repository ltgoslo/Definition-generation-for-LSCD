## Definition Generation for Lexical Semantic Change Detection (LSCD)

This repository contains the code and some data for the paper [Definition generation for lexical semantic change detection](https://arxiv.org/abs/2406.14167) by Mariia Fedorova, Andrey Kutuzov and Yves Scherrer.

# The work to clean up the code is ongoing

## Repository structure
    .
    ├── apd_prt                 # APD and PRT experiments
    ├── definition_generation   # generating definitions
    ├── embeddings              # generating definitions' embeddings
    ├── generated_definitions   # prompts and definitions generated
    ├── src                     # Running Tang et al. (2023)'s method, see more in its README

## Obtain the data

### Lists of words and ground truth

```src/data/``` 

### Diachronical corpora

Sampled usage examples with prompts and generated definitions can be found in ```generated_definitions/```.

The usage examples were sampled from the following resources:

- [English](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/)
- Norwegian: [NBDigital corpus](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-34/) and [Norsk aviskorpus](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/) (available under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/))
- [Russian](https://ruscorpora.ru/new/en/corpora-usage.html); the corpora's license does not allow publishing them; for that reason, we could only release the prompts and definitions without usage examples. Any other corpus may be used instead of it (although the results may be different then).

## [Reproduce the baselines](https://github.com/ltgoslo/Definition-generation-for-LSCD/tree/main/src#reproduce-lesk-baselines) (Table 2)

## Definition generation and evaluation

```commandline
cd definition_generation
git clone git@github.com:ltgoslo/definition_modeling.git
./definition_generation_pipeline.sh ${}
```
Read about the generation parameters in the [README file](definition_generation/README.md).

## Reproduce evaluation of LSCD performance with definition embeddings obtained with different decoding strategies (Table 3)

```commandline
cd apd_prt
./evaluate.sh
```

## Reproduce evaluation of LSCD performance with merged definitions obtained with different decoding strategies (Table 4)

```commandline
./merge_all.sh
```

## Reproduce Figure 1

`src/analysis/graphs.ipynb`