#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python2 src/scripts/babble_reduce.py \
  --data-dir ./data \
  --out-dir ./output \
  --save-reduced-dir ./data/babble \
  --dataset-id cdr \
  --fit-intercept \
  --sample-weights

python2 src/scripts/load_all_datasets.py \
  --data-dir ./data \
  --dataset-ids cdr reduced_cdr \
  --supplement-ids \
  cdr_LFs \
  reduced_cdr_LFs \
  cdr_nonfires \
  reduced_cdr_nonfires \
  cdr_weights \
  reduced_cdr_weights \
  cdr_labeling_info \
  reduced_cdr_labeling_info
