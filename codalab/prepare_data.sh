#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
DATASET_IDS=$@
echo $DATASET_IDS

python2 src/scripts/load_all_datasets.py \
  --data-dir ./data \
  --dataset-ids $DATASET_IDS \
  --supplement-ids none
