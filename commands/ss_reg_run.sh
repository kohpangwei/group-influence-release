#!/bin/bash

task_index=$1

JUICE_HOME=/juice/u/hteo/
source $HOME/envs/infl2/bin/activate
INFL_HOME=$HOME/code/influence
#INFL_HOME=$HOME/code/dev_influence
export PYTHONPATH=$INFL_HOME
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

DATA_DIR=/u/nlp/influence/data/
OUT_DIR=/u/nlp/influence/output_arxiv/

rel_reg_index=$(($task_index % 10))
dataset_index=$(($task_index / 10))

dataset_ids=(hospital spam mnist_small dogfish)
dataset_id=${dataset_ids[$dataset_index]}

rel_regs=(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 1e1 1e2)

reg=${rel_regs[$rel_reg_index]}
subset_min_rel_size=0.0025
subset_max_rel_size=0.25
num_subsets=10

args=""

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
        --no-intermediate-results \
    --fixed-reg $reg
