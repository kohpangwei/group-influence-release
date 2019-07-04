#!/bin/bash

task_index=$1

JUICE_HOME=/juice/u/hteo/
source $HOME/envs/infl2/bin/activate
INFL_HOME=$HOME/code/influence
#INFL_HOME=$HOME/code/dev_influence
export PYTHONPATH=$INFL_HOME
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

DATA_DIR=/u/nlp/influence/data_codalab/
OUT_DIR=/u/nlp/influence/output_codalab/
SAVE_REDUCED_DIR=/u/nlp/influence/data_codalab/babble/

args=""

python $INFL_HOME/scripts/babble_reduce.py \
    $args \
        --data-dir $DATA_DIR \
        --out-dir $OUT_DIR \
        --save-reduced-dir $SAVE_REDUCED_DIR \
        --dataset-id cdr \
        --fit-intercept \
        --sample-weights

python $INFL_HOME/scripts/data_valuation.py \
    $args \
        --data-dir $DATA_DIR \
        --out-dir $OUT_DIR \
        --dataset-id reduced_cdr \
        --sample-weights

python $INFL_HOME/scripts/credit_assignment.py \
    $args \
        --data-dir $DATA_DIR \
        --out-dir $OUT_DIR \
        --dataset-id mnli \
        --no-intermediate-results
