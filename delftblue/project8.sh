#!/bin/bash

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate project8
cd ..
AMGX_CONFIG_PATH=$HOME/.local/lib/configs python -m project8 $(< delftblue/args.txt)
