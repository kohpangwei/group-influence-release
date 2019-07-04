#!/bin/bash

# NOTE: WE RECOMMEND 32GB OF RAM FOR THIS RUN.

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

dataset_index=$task_index

dataset_ids=(hospital spam mnist_small dogfish animals mnist cifar10 cifar10_small reduced_cdr)
dataset_id=${dataset_ids[$dataset_index]}
subset_min_rel_size=0.0025
subset_max_rel_size=0.25
num_subsets=100

if [[ "$dataset_id" == "cifar10" ]]; then
    num_subsets=10
elif [[ "$dataset_id" == "cifar10_small" ]]; then
    num_subsets=25
fi

args=""

if [[ "$2" == "--worker" ]]; then
    args="--worker"
fi

python $INFL_HOME/scripts/subsets_infl_logreg.py \
    $args \
    --data-dir $DATA_DIR \
    --out-dir $OUT_DIR \
    --dataset-id $dataset_id \
    --subset-choice-type range \
    --subset-min-rel-size $subset_min_rel_size \
    --subset-max-rel-size $subset_max_rel_size \
    --max-memory 100000000 \
    --num-subsets $num_subsets \
    --no-intermediate-results

