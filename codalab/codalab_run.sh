#!/bin/bash

# This script should be run from the root directory of the repository and
# contains all the commands used to prepare the Codalab worksheet.

CL_RUN_FLAGS="--request-memory 32g --request-gpus 1 --request-docker-image bbbert/influence:5"

# Upload source code
cl upload -n src-datasets datasets/*.py
cl upload -n src-experiments experiments/*.py
cl upload -n src-influence influence/*.py
cl upload -n src-scripts scripts/*.py
cl make -n src datasets:src-datasets experiments:src-experiments influence:src-influence scripts:src-scripts

# Upload Codalab scripts
cl upload -n cl codalab/*.sh

# Upload figures notebook
cl upload paper-figures.ipynb

# Prepare all datasets except for MNLI, which is uploaded pre-processed for now
cl run -n data-babble $CL_RUN_FLAGS --request-network :src :cl "bash cl/prepare_babble.sh"
cl run -n data-hospital $CL_RUN_FLAGS --request-network :src :cl "bash cl/prepare_data.sh hospital"
cl run -n data-mnist $CL_RUN_FLAGS --request-network :src :cl "bash cl/prepare_data.sh mnist mnist_small"
cl run -n data-animals $CL_RUN_FLAGS --request-network :src :cl "bash cl/prepare_data.sh dogfish animals"
cl run -n data-spam $CL_RUN_FLAGS --request-network :src :cl "bash cl/prepare_data.sh spam"

# Run the group influence experiments
cl run -n subsets-reduced-cdr $CL_RUN_FLAGS :src :cl :data-babble "bash cl/link_data.sh; bash cl/run_subsets.sh reduced_cdr"
cl run -n subsets-hospital $CL_RUN_FLAGS :src :cl :data-hospital "bash cl/link_data.sh; bash cl/run_subsets.sh hospital"
cl run -n subsets-mnist $CL_RUN_FLAGS :src :cl :data-mnist "bash cl/link_data.sh; bash cl/run_subsets.sh mnist"
cl run -n subsets-mnist-small $CL_RUN_FLAGS :src :cl :data-mnist "bash cl/link_data.sh; bash cl/run_subsets.sh mnist_small"
cl run -n subsets-animals $CL_RUN_FLAGS :src :cl :data-animals "bash cl/link_data.sh; bash cl/run_subsets.sh animals"
cl run -n subsets-dogfish $CL_RUN_FLAGS :src :cl :data-animals "bash cl/link_data.sh; bash cl/run_subsets.sh dogfish"
cl run -n subsets-spam $CL_RUN_FLAGS :src :cl :data-spam "bash cl/link_data.sh; bash cl/run_subsets.sh spam"

# Run the regularization experiments
cl run -n reg-hospital $CL_RUN_FLAGS :src :cl :data-hospital "bash cl/link_data.sh; bash cl/run_reg.sh hospital"
cl run -n reg-mnist-small $CL_RUN_FLAGS :src :cl :data-mnist "bash cl/link_data.sh; bash cl/run_reg.sh mnist_small"
cl run -n reg-dogfish $CL_RUN_FLAGS :src :cl :data-animals "bash cl/link_data.sh; bash cl/run_reg.sh dogfish"
cl run -n reg-spam $CL_RUN_FLAGS :src :cl :data-spam "bash cl/link_data.sh; bash cl/run_reg.sh spam"

# Construct counterexamples
cl run -n counterexamples $CL_RUN_FLAGS :src :cl "bash cl/run_counterexamples.sh"

# Run applications
cl run -n data-valuation $CL_RUN_FLAGS :src :cl :data-babble "bash cl/link_data.sh; bash cl/run_data_valuation.sh"
cl run -n credit-assignment $CL_RUN_FLAGS :src :cl :data-mnli "bash cl/link_data.sh; bash cl/run_credit_assignment.sh"

# Plot figures
cl run -n figures $CL_RUN_FLAGS :src :cl :paper-figures.ipynb \
  :data-babble \
  :data-hospital \
  :data-mnist \
  :data-mnli \
  :data-animals \
  :data-spam \
  :subsets-reduced-cdr \
  :subsets-hospital \
  :subsets-mnist \
  :subsets-mnist-small \
  :subsets-animals \
  :subsets-dogfish \
  :subsets-spam \
  :reg-hospital \
  :reg-mnist-small \
  :reg-dogfish \
  :reg-spam \
  :counterexamples \
  :data-valuation \
  :credit-assignment \
  "bash cl/link_data.sh; bash cl/link_output.sh; bash cl/run_figures.sh"
