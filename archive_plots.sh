#!/bin/bash

# Check if an argument was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <archive_folder>"
    echo "Example: $0 archivefoldername/"
    exit 1
fi

# First argument
destination="${1}"

mkdir -p ${destination}
cp -r compression/ ${destination}
cp -r matrix_internode/ ${destination}
cp -r rank_internode/ ${destination}
cp -r overall/ ${destination}
