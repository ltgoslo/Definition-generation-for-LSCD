#!/bin/bash

#SBATCH --job-name=embed_defs
#SBATCH --account=nn9851k
#SBATCH --time=6:00:00
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

module load nlpl-scipy-ecosystem/01-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8

IN=${1}
OUT=${2}

python embed_definitions.py -m sentence-transformers/distiluse-base-multilingual-cased-v2 -i $IN -b 8 -o $OUT.npz
