#!/bin/bash

echo $SHELL
eval "$(conda shell.bash hook)"
conda activate ftc310
cd /work/marioeduardo-a/github/ftc-mario
python ftc_dataset.py
