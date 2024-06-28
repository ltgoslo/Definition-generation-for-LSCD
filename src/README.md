# Instructions  

The **bayesian_optimization.py** and **sense_dis.py** scripts were taken from [Tang et al. (2023)](https://aclanthology.org/2023.findings-emnlp.231/) [repository](https://github.com/LivNLP/Sense-based-Semantic-Change-Prediction) and modified by us.

## Folder structure

    .
    ├── analysis/                   # results and figure
    ├── data/                       # ground truth
    ├── logs/                       # evaluation logs
    ├── sampling_usages/            # sample usage examples for definition generation
    ├── bayesian_optimization.py    # Tang et al. (2023) classification (not used by us, modified for better reproducability).
    ├── eval.py                     # SemEval official Evaluation script
    ├── eval.sh                     # running eval.py  
    ├── merge_definitions_string.py # merge definitions
    ├── norwegian-dataset.ipynb     # parse Bokmålsordboka
    ├── rm_prompts.py               # remove prompts or their parts from data (e.g. for data publication)
    ├── sense_dis.py                # ranking by Lesk
    ├── utils.py                    # ranking utils

## Reproducing Lesk baselines

### Input data

#### Bokmålsordboka and Wiktionary instead of WordNet for Norwegian and Russian

- in order to get Bokmålsordboka in XML format, contact its [developers](https://ordbokene.no/eng/contact). You will have to [parse](https://github.com/ltgoslo/Definition-generation-for-LSCD/blob/main/src/norwegian-dataset.ipynb) it into `complete.tsv.gz` in CODWOE format (the same as for Russian) yourself. Put it into `data/norwegian1/` and `data/norwegian2/`

- for Russian Wiktionary, download `Full datasets` from [CODWOE](https://codwoe.atilf.fr/). Extract `ru.complete.csv` and put it into `data/russian1/`, `data/russian2/` and `data/russian3/` .

### Running Lesk algorithm with part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --use_pos_in_lesk True --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```

### Running Lesk algorithm without part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```

## Warning

1. Different samples of usage examples may result in different output scores. We pushed exact samples we used for English and Norwegians into `predictions/lesk/<language>` (the files `sent_ls1.json` and `sent_ls2.json`). These files for English and Norwegian 2 are reproducible (if you remove them and run `sense_dis.py` as described above, it will recreate them). For Norwegian 1, some sentences sampled will be different and we have not found the reason so far. But Lesk results for Norwegian 1 are not statistically significant with any sample of usages, so we decided not to care much. 

2. All Russian target words are not PoS-ambiguous, so the results will be the same.

### Evaluating Lesk algorithm

Running the official SemEval evaluation script:

```commandline
python eval.py 2 predictions/lesk/<language>/<metric>_dict.tsv data/<language>/truth/graded.txt
```

## Reproducing Tang et al. (2023) NLTK accuracy (Table 1 in their paper)

### Warning

They use ax-platform, which requires [PyTorch](https://pytorch.org/get-started/locally/) to be installed

```commandline
python bayesian_optimization.py --sense_dis_method_dir predictions/lesk/english --method ax
```
