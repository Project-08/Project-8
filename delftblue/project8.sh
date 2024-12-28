#!/bin/bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate project8
cd ..
AMGX_CONFIG_PATH=$HOME/.local/lib/configs python -m project8 bench solve-tri --min 1000 --plot --attempts 3 --max 10000 --steps 4 --solvers cuSPARSE AmgX
