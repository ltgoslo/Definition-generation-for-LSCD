# Instructions  

The files **bayesian_optimization.py** and **sense_dis.py** were taken from [Tang et al. (2023)](https://aclanthology.org/2023.findings-emnlp.231/) [repository](https://github.com/LivNLP/Sense-based-Semantic-Change-Prediction) and modified by us.

## Folder structure

    .
    ├── data                       # ground truth
    ├── logs                       # evaluation logs
    ├── bayesian_optimization.py   # Tang et al. (2023) classification (not used by us, modified for better reproducability).
    ├── eval.py                    # SemEval official Evaluation script
    ├── eval.sh                    # running eval.py  
    ├── sense_dis.py               # ranking by Lesk
    ├── utils.py                   # ranking utils

## Reproduce Lesk baselines

### Input data

#### Diachronic corpora
Obtain the English corpora as described in [README](https://github.com/ltgoslo/Definition-generation-for-LSCD?tab=readme-ov-file#obtain-the-data) and put them into `data/english/`. They are expected to be named `corpus1/ccoha1.txt` and `corpus2/ccoha2.txt`.

#### Bokmålsordboka and Wiktionary instead of WordNet for Norwegian and Russian

- in order to get Bokmålsordboka in XML format, contact its [developers](https://ordbokene.no/eng/contact). You will have to [parse](https://github.com/ltgoslo/Definition-generation-for-LSCD/blob/main/src/norwegian-dataset.ipynb) it into `complete.tsv.gz` in CODWOE format (the same as for Russian) yourself. Put it into `data/norwegian1/` and `data/norwegian2/`

- for Russian Wiktionary, download `Full datasets` from [CODWOE](https://codwoe.atilf.fr/). Extract `ru.complete.csv` and put it into `data/russian1/`, `data/russian2/` and `data/russian3/` .

### Run Lesk with part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --use_pos_in_lesk True --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```

### Run Lesk without part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```
