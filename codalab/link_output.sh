#!/bin/bash

# This script collates experiment results from the output/ subfolder in
# multiple codalab bundles into a single output/ subfolder, via symlinks.

mkdir -p output

EXPERIMENT_RUNS=(
  subsets-animals
  subsets-dogfish
  subsets-hospital
  subsets-mnist
  subsets-mnist-small
  subsets-reduced-cdr
  subsets-spam
  reg-dogfish
  reg-hospital
  reg-mnist-small
  reg-spam
  counterexamples
  data-valuation
  credit-assignment
)

for folder in "${EXPERIMENT_RUNS[@]}"; do
  for experiment_folder in $folder/output/*; do
    EXPERIMENT_ID=${experiment_folder##*/output/}
    mkdir -p output/$EXPERIMENT_ID
    find $folder/output/$EXPERIMENT_ID -mindepth 1 -maxdepth 1 -type d -exec ln -s ../../'{}' output/${EXPERIMENT_ID}/ \;
  done
done
