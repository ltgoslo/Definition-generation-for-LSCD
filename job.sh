#!/bin/bash

# Job name:
#SBATCH --job-name=run_defgen
#
# Project:
#SBATCH --account=nn9851k
#
# Wall time limit:
#SBATCH --time=00-20:00:00

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default

module load nlpl-nlptools/01-foss-2022b-Python-3.10.8

module list    # For easier debugging

REPO=/cluster/projects/nn9851k/andreku/defgen_lscd/Sense-based-Semantic-Change-Prediction/
DATA_PATH=/cluster/projects/nn9851k/corpora/diachronic/acl_data
RES_PATH=/cluster/projects/nn9851k/corpora/diachronic/acl_results
MODELS_DIR=/cluster/projects/nn9851k/models/definition_generation
python "${REPO}run_defgen.py" --data_path $DATA_PATH --bsize 8 --res_path $RES_PATH --n_first 16 --models_dir $MODELS_DIR