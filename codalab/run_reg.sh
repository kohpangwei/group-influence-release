#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
DATASET_ID=$1
REL_REGS="1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 1e1 1e2"

for REG in $REL_REGS; do
  python2 src/scripts/subsets_infl_logreg.py \
    --data-dir ./data \
    --out-dir ./output \
    --dataset-id $DATASET_ID \
    --subset-choice-type range \
    --subset-min-rel-size 0.0025 \
    --subset-max-rel-size 0.25 \
    --num-subsets 10 \
    --max-memory 100000000 \
    --no-intermediate-results \
    --fixed-reg $REG
done
