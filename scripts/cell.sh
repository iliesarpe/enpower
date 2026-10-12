#!/bin/bash
# One unit of work (a "cell") of one experiment, run in the same way on the
# local machine and in a SLURM job array:
#
#   scripts/cell.sh <experiment> count    the number of cells
#   scripts/cell.sh <experiment> <i>      run cell i (0-based)
#
# experiment   one cell is
#   splits     (seed, dataset, ensemble): train the ensemble of the split and run
#              Tr-PS and Tr-MLE (python/train_binary.py), then ENPOWER_B, WMG, ISCG,
#              MC-beta and MC-phi (C++)
#   mc         (seed, dataset, ensemble): train the ensemble on the multiclass
#              labels (python/train_multiclass.py), then measure the runtime of
#              ENPOWER_MC-beta, ENPOWER_MC-phi and ENPOWER_MC-phi++ (C++)
#   clip-votes (dataset): CLIP inference, i.e., the votes of the 22 prompts on
#              every image, and the power indices of the Python reference
#              implementation on split 100
#   clip-run   (seed, dataset): Acc, ENPOWER_MC-beta, ENPOWER_MC-phi, Tr-PS and
#              Tr-MLE on D_val
#   synth      shard i of SYNTH_SHARDS of the synthetic grid SYNTH_CONFIG
#              (requires a previous run of `enpower.py synth gen`)
#
# Every step whose output exists is skipped, so rerunning a cell resumes it.
# CELLS_FILTER restricts the cells, e.g., CELLS_FILTER="--seeds 100 --datasets credit-g".
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/common.sh"

EXP=$1
I=$2
SYNTH_CONFIG=${SYNTH_CONFIG:-paper}
SYNTH_SHARDS=${SYNTH_SHARDS:-16}

# one thread per cell, so that no timed method shares its core with another
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cells() {   # the cells of an experiment, one per line
   case $1 in
      splits) enp $PY $ENP splits cells ${CELLS_FILTER} ;;
      mc) enp $PY $ENP mc cells ${CELLS_FILTER} ;;
      clip-votes) enp $PY $ENP clip cells ${CELLS_FILTER} | awk '!seen[$2]++ {print $2}' ;;
      clip-run) enp $PY $ENP clip cells ${CELLS_FILTER} ;;
      synth) seq 0 $((SYNTH_SHARDS - 1)) ;;
      *) echo "unknown experiment: $1" >&2; exit 2 ;;
   esac
}

if [ "$I" = "count" ]; then
   cells "$EXP" | wc -l | tr -d ' '
   exit 0
fi
LINE=$(cells "$EXP" | sed -n "$((I + 1))p")
[ -n "$LINE" ] || { echo "$EXP: no cell $I" >&2; exit 2; }
echo "[$EXP] cell $I: $LINE"

case $EXP in
   splits)
      read SEED DATASET ID ENSEMBLE <<< "$LINE"
      OUT=outputs/splits/seed${SEED}
      if [ ! -f "$REPO/$OUT/baselines/binary_hpc_${ID}_${ENSEMBLE}.json" ]; then
         enp $PY python/train_binary.py --dataset_id "$ID" --seed "$SEED" \
             --out-root "$OUT" --ensembles "$ENSEMBLE"
      fi
      enp $PY $ENP splits run --seeds "$SEED" --datasets "$DATASET" --ensembles "$ENSEMBLE"
      enp $PY $ENP splits mc --seeds "$SEED" --datasets "$DATASET" --ensembles "$ENSEMBLE"
      ;;
   mc)
      read SEED DATASET ID ENSEMBLE <<< "$LINE"
      OUT=outputs/mc_splits/seed${SEED}
      if [ ! -f "$REPO/$OUT/ensembles_multiclass/${DATASET}_${ENSEMBLE}_predictions.npz" ]; then
         enp $PY python/train_multiclass.py --dataset_id "$ID" --seed "$SEED" \
             --out-root "$OUT" --ensembles "$ENSEMBLE"
      fi
      enp $PY $ENP mc run --seeds "$SEED" --datasets "$DATASET" --ensembles "$ENSEMBLE"
      ;;
   clip-votes)
      DATASET=$LINE
      # inference uses all the cores of the cell
      export OMP_NUM_THREADS=${CLIP_THREADS:-${SLURM_CPUS_PER_TASK:-4}} MKL_NUM_THREADS=${CLIP_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
      enp $PY $ENP clip votes --datasets "$DATASET"
      if [ ! -f "$REPO/outputs/${DATASET}_ViT-B-32_results.json" ]; then
         enp $PY python/clip_inference.py --dataset "$DATASET" --device cpu
      fi
      ;;
   clip-run)
      read SEED DATASET <<< "$LINE"
      enp $PY $ENP clip run --seeds "$SEED" --datasets "$DATASET"
      ;;
   synth)
      enp $PY $ENP synth run --config "$SYNTH_CONFIG" --shard "$LINE" --shards "$SYNTH_SHARDS"
      ;;
esac
