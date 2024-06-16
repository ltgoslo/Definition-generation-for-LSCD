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

Obtain the corpora as described in [README](https://github.com/ltgoslo/Definition-generation-for-LSCD?tab=readme-ov-file#obtain-the-data) and put them into the corresponding language subfolders in data/.

### Run Lesk with part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --use_pos_in_lesk True --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```

### Run Lesk without part-of-speech tags

```commandline
python sense_dis.py --data_dir data/ --results_dir predictions/ --lang <english|norwegian1|norwegian2|russian1|russian2|russian3>
```

### Evaluate the results

```commandline
./eval.sh  <english|norwegian1|norwegian2|russian1|russian2|russian3> 2 predictions/
```