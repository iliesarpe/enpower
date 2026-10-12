#!/bin/bash
#SBATCH --job-name=enpower-mc
#SBATCH --output logs/mc_%A_%a.txt
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00

# Runtime for multiclass classification (Table 5) over five random splits.
# From the root of the repository:
#
#   mkdir -p logs && sbatch slurm/mc.slurm      (or ./reproduce.sh submit mc)
#
# This job builds the C++ library, downloads the tabular datasets, and submits
# itself again as a job array with one job per (seed, dataset, ensemble), with
# at most MAX_RUNNING jobs running concurrently. Each array job runs scripts/cell.sh mc <i>.
# Set SBATCH_PARTITION (and SBATCH_ACCOUNT) for the cluster.
set -e
REPO=${SLURM_SUBMIT_DIR}
cd "$REPO"
source scripts/common.sh

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
   enp make -C cpp clean
   enp make -C cpp all
   enp $PY python/download_data.py --only tabular
   N=$(scripts/cell.sh mc count)
   sbatch --array=0-$((N - 1))%${MAX_RUNNING:-16} slurm/mc.slurm
else
   scripts/cell.sh mc "${SLURM_ARRAY_TASK_ID}"
fi
