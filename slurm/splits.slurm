#!/bin/bash
#SBATCH --job-name=enpower-splits
#SBATCH --output logs/splits_%A_%a.txt
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00

# Binary classification experiments (Sec. 5.1-5.3) over five random splits.
# From the root of the repository:
#
#   mkdir -p logs && sbatch slurm/splits.slurm      (or ./reproduce.sh submit splits)
#
# This job builds the C++ library, downloads the tabular datasets (so that
# concurrent jobs do not write to the cache at the same time), and submits
# itself again as a job array with one job per (seed, dataset, ensemble), with
# at most MAX_RUNNING jobs running concurrently. Each job of the array runs
# scripts/cell.sh splits <i>. Set SBATCH_PARTITION (and SBATCH_ACCOUNT) for the
# cluster.
set -e
REPO=${SLURM_SUBMIT_DIR}
cd "$REPO"
source scripts/common.sh

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
   enp make -C cpp clean
   enp make -C cpp all
   enp $PY python/download_data.py --only tabular
   N=$(scripts/cell.sh splits count)
   sbatch --array=0-$((N - 1))%${MAX_RUNNING:-16} slurm/splits.slurm
else
   scripts/cell.sh splits "${SLURM_ARRAY_TASK_ID}"
fi
