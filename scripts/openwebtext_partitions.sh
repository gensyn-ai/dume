#!/bin/bash

mkdir -p train validation test

for file in data_*.parquet; do

    x=$(echo "$file" | sed -E 's/data_([0-9]+)_[0-9]+\.parquet/\1/')

    if [[ $x -ge 0 && $x -le 20 ]]; then
        mv "$file" train/
    elif [[ $x -ge 21 && $x -le 24 ]]; then
        mv "$file" validation/
    else
        mv "$file" test/
    fi
    
done