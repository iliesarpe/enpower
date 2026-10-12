#!/bin/bash
#SBATCH --job-name=enpower-synth
#SBATCH --output logs/synth_%A_%a.txt
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02-00:00

# Synthetic experiments (Appendix G.6). From the root of the repository:
#
#   mkdir -p logs && sbatch slurm/synth.slurm      (or ./reproduce.sh submit synth)
#
# This job builds the C++ library, generates the deterministic instances of
# SYNTH_CONFIG (default: paper), and submits itself again as a job array of
# SYNTH_SHARDS (default: 16) jobs, each running one part of the grid
# (scripts/cell.sh synth <i>). All values of n of a (task, L, |U|, p) are
# assigned to the same shard, so that after a timeout the larger values of n
# are skipped. Resubmitting with the same SYNTH_SHARDS resumes the grid.
# Set SBATCH_PARTITION (and SBATCH_ACCOUNT) for the cluster.
set -e
REPO=${SLURM_SUBMIT_DIR}
cd "$REPO"
source scripts/common.sh
export SYNTH_CONFIG=${SYNTH_CONFIG:-paper} SYNTH_SHARDS=${SYNTH_SHARDS:-16}

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
   enp make -C cpp clean
   enp make -C cpp all
   enp $PY $ENP synth gen --config "$SYNTH_CONFIG"
   sbatch --export=ALL --array=0-$((SYNTH_SHARDS - 1)) slurm/synth.slurm
else
   scripts/cell.sh synth "${SLURM_ARRAY_TASK_ID}"
fi
