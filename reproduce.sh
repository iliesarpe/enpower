#!/bin/bash
# Reproduces the experiments of the paper. See REPRODUCE.md.
#
#   ./reproduce.sh all                 container, build, data, check, run all, report all
#   ./reproduce.sh container           build the Apptainer image enpower.sif
#   ./reproduce.sh build               compile the C++ library (cpp/solver, cpp/check)
#   ./reproduce.sh data [--only tabular|clip]   download the datasets and CLIP into data/
#   ./reproduce.sh check [ARGS]        test the C++ algorithms against brute force (cpp/check)
#   ./reproduce.sh smoke               a small end-to-end run to test the setup
#   ./reproduce.sh run    EXP|all      all the cells of EXP, sequentially, on the local machine
#   ./reproduce.sh submit EXP|all      the cells of EXP as SLURM job arrays
#   ./reproduce.sh report EXP|all [ARGS]   figures and tables, written to results/
#
#   EXP is one of splits, mc, clip, synth (and, for report, tables).
#
# Every command runs in the image (ENPOWER_SIF, default ./enpower.sif); set
# ENPOWER_NO_CONTAINER=1 (and PYTHON) to run on the host instead.
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"
source scripts/common.sh

EXPERIMENTS="splits mc clip synth"
export SYNTH_CONFIG=${SYNTH_CONFIG:-paper}

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

each() {   # each EXP: run all the cells of one experiment on the local machine
   local exp=$1 n i
   n=$(scripts/cell.sh "$exp" count)
   for ((i = 0; i < n; i++)); do scripts/cell.sh "$exp" "$i"; done
}

run() {
   case $1 in
      splits|mc) each "$1" ;;
      clip) each clip-votes; each clip-run ;;
      synth)
         enp $PY $ENP synth gen --config "$SYNTH_CONFIG"
         SYNTH_SHARDS=1 each synth ;;
      all) for e in $EXPERIMENTS; do run "$e"; done ;;
      *) usage ;;
   esac
}

submit() {
   mkdir -p logs
   case $1 in
      splits|mc|clip|synth) sbatch "slurm/$1.slurm" ;;
      all) for e in $EXPERIMENTS; do submit "$e"; done ;;
      *) usage ;;
   esac
}

report() {
   local exp=$1; shift || true
   case $exp in
      splits|mc) enp $PY $ENP "$exp" report "$@" ;;
      clip)
         enp $PY $ENP clip report "$@"
         for d in cifar100 imagenet-a pets; do
            if [ -f "outputs/${d}_ViT-B-32_results.json" ]; then
               enp $PY python/clip_prompt_table.py "outputs/${d}_ViT-B-32_results.json" results/tables
            fi
         done ;;
      synth) enp $PY $ENP synth report --config "$SYNTH_CONFIG" "$@" ;;
      tables) enp $PY python/dataset_table.py "$@" ;;
      all) for e in $EXPERIMENTS tables; do report "$e"; done ;;
      *) usage ;;
   esac
}

build() {
   enp make -C cpp clean
   enp make -C cpp all check
}

smoke() {
   # the steps of the full run, on two small datasets, one cell each
   build
   enp cpp/check --nmax 6 --Lmax 4 --trials 200 --rnmax 20 --rLmax 5 --bnmax 60
   enp $PY python/download_data.py --only tabular --openml-ids 31 1468
   local splits_cell="--seeds 100 --datasets credit-g --ensembles random_forest"
   local mc_cell="--seeds 100 --datasets cnae-9 --ensembles random_forest"
   CELLS_FILTER=$splits_cell scripts/cell.sh splits 0
   CELLS_FILTER=$mc_cell scripts/cell.sh mc 0
   enp $PY $ENP synth gen --config small
   SYNTH_CONFIG=small SYNTH_SHARDS=1 scripts/cell.sh synth 0
   enp $PY $ENP splits report $splits_cell --out results/smoke/splits
   enp $PY $ENP mc report $mc_cell --out results/smoke/mc
   enp $PY $ENP synth report --config small --out results/smoke/synth
   echo "smoke run complete: results/smoke/"
}

cmd=${1:-}; shift || true
case $cmd in
   container) apptainer build --fakeroot "$IMAGE" container/enpower.def ;;
   build) build ;;
   data) enp $PY python/download_data.py "$@" ;;
   check) enp cpp/check "$@" ;;
   smoke) smoke ;;
   run) run "${1:-all}" ;;
   submit) submit "${1:-all}" ;;
   report) report "${1:-all}" "${@:2}" ;;
   all)
      [ -f "$IMAGE" ] || [ "${ENPOWER_NO_CONTAINER:-0}" = "1" ] || apptainer build --fakeroot "$IMAGE" container/enpower.def
      build
      enp $PY python/download_data.py
      enp cpp/check
      run all
      report all ;;
   *) usage ;;
esac
