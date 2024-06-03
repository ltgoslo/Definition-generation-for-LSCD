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
    ├── sense_dis_defgen.py        # ranking by definitions as senses
    ├── sense_dis_defgen_merged.py # ranking by merged definitions as senses
    ├── utils.py                   # ranking utils
