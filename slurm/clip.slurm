#!/bin/bash
#SBATCH --job-name=enpower-clip
#SBATCH --output logs/clip_%A_%a.txt
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00

# Prompt ensembling of CLIP (Sec. 5.4) over five random splits.
# From the root of the repository:
#
#   mkdir -p logs && sbatch slurm/clip.slurm      (or ./reproduce.sh submit clip)
#
# This job builds the C++ library, downloads the image datasets and CLIP, and
# submits two job arrays:
#   STAGE=votes  one job per dataset on 8 CPUs: CLIP inference (scripts/cell.sh clip-votes <i>);
#   STAGE=run    one job per (seed, dataset) on one CPU, after all the jobs of
#                STAGE=votes have succeeded (scripts/cell.sh clip-run <i>).
# Set SBATCH_PARTITION (and SBATCH_ACCOUNT) for the cluster.
set -e
REPO=${SLURM_SUBMIT_DIR}
cd "$REPO"
source scripts/common.sh
VOTES_CPUS=8

if [ -z "${STAGE}" ]; then
   enp make -C cpp clean
   enp make -C cpp all
   enp $PY python/download_data.py --only clip
   N=$(scripts/cell.sh clip-votes count)
   VOTES=$(sbatch --parsable --export=ALL,STAGE=votes --cpus-per-task=${VOTES_CPUS} \
           --time=01-00:00 --array=0-$((N - 1)) slurm/clip.slurm)
   N=$(scripts/cell.sh clip-run count)
   sbatch --export=ALL,STAGE=run --dependency=afterok:${VOTES} \
          --array=0-$((N - 1))%${MAX_RUNNING:-16} slurm/clip.slurm
elif [ "${STAGE}" = "votes" ]; then
   scripts/cell.sh clip-votes "${SLURM_ARRAY_TASK_ID}"
else
   scripts/cell.sh clip-run "${SLURM_ARRAY_TASK_ID}"
fi
