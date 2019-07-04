#!/bin/bash

task_index=$1

JUICE_HOME=/juice/u/hteo/
source $HOME/envs/infl2/bin/activate
INFL_HOME=$HOME/code/influence
#INFL_HOME=$HOME/code/dev_influence
export PYTHONPATH=$INFL_HOME
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

OUT_DIR=/u/nlp/influence/output_codalab/

dataset_index=$task_index

dataset_ids=(ortho2 gauss4)
dataset_id=${dataset_ids[$dataset_index]}

args=""

python $INFL_HOME/scripts/counterexamples.py \
    $args \
        --out-dir $OUT_DIR \
        --dataset-id $dataset_id \
        --no-intermediate-results

