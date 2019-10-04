#!/bin/bash

# This script collates preprocessed datasets from the data/ subfolder in
# multiple codalab bundles into a single data/ subfolder, via symlinks.
# It also handles MNLI via a special case, since we upload it directly.

mkdir -p data

find . -mindepth 1 -maxdepth 1 -type l -name 'data-*' | while read folder; do
  if [[ "$folder" == "./data-mnli" ]]; then
    ln -s ../data-mnli ./data/multinli_1.0
  else
    find $folder/data -mindepth 1 -maxdepth 1 -type d -exec ln -s ../'{}' data/ \;
  fi
done
