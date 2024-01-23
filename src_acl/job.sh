#!/bin/bash

# Job name:
#SBATCH --job-name=run_defgen
#
# Project:
#SBATCH --account=nn9851k
#
# Wall time limit:
#SBATCH --time=4:00:00
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8


module list    # For easier debugging
export PYTHONIOENCODING=utf8
REPO=/cluster/projects/nn9851k/andreku/defgen_lscd/Sense-based-Semantic-Change-Prediction/
DATA_PATH=/cluster/projects/nn9851k/andreku/defgen_lscd/acl_data
RES_PATH=/cluster/projects/nn9851k/andreku/defgen_lscd/acl_results
MODELS_DIR=/cluster/projects/nn9851k/models/definition_generation
BATCH_SIZE=${1}
MAX_NEW_TOKENS=${2}
N_FIRST=${3}
python "${REPO}run_defgen.py" --data_path $DATA_PATH --bsize $BATCH_SIZE --res_path $RES_PATH --n_first $N_FIRST --models_dir $MODELS_DIR --max_new_tokens $MAX_NEW_TOKENS