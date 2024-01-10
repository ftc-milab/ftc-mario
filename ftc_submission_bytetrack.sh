#!/bin/bash

echo $SHELL
eval "$(conda shell.bash hook)"
conda activate ftc310
echo $(which python)
cd /work/marioeduardo-a/github/ftc-mario
pwd
python ftc_submission_bytetrack.py
