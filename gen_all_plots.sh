#!/bin/bash

# Check if an argument was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_folder>"
    echo "Example: $0 input_folder/"
    exit 1
fi

# First argument
input_folder="$1"

. .venv/bin/activate

python3 parse.py ${input_folder} internode.csv

python3 plot.py
python3 merge_plots.py
