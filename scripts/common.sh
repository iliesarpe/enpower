# Sourced by reproduce.sh, scripts/cell.sh and slurm/*.slurm, with REPO set to
# the root of the repository. Defines enp, which runs a command from the root
# of the repository inside the Apptainer image (or on the host when
# ENPOWER_NO_CONTAINER=1), and PY, the Python interpreter to call through it.
#
#   ENPOWER_SIF            the image (default: <repo>/enpower.sif)
#   ENPOWER_NO_CONTAINER   1 to run on the host instead of in the image
#   PYTHON                 the host interpreter when ENPOWER_NO_CONTAINER=1 (default: python3)

: "${REPO:?REPO must be set to the root of the repository}"
IMAGE=${ENPOWER_SIF:-$REPO/enpower.sif}
ENP=cpp/tools/enpower.py

if [ "${ENPOWER_NO_CONTAINER:-0}" = "1" ]; then
   PY=${PYTHON:-python3}
   enp() { (cd "$REPO" && SCIKIT_LEARN_DATA="$REPO/data/openml" "$@"); }
else
   PY=python
   enp() {
      if [ ! -f "$IMAGE" ]; then
         echo "no image at $IMAGE: run ./reproduce.sh container, set ENPOWER_SIF," \
              "or set ENPOWER_NO_CONTAINER=1 to run on the host" >&2
         return 1
      fi
      APPTAINERENV_SCIKIT_LEARN_DATA=/workspace/data/openml \
         apptainer exec --bind "$REPO:/workspace" --pwd /workspace "$IMAGE" "$@"
   }
fi
