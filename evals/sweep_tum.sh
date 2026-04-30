#!/bin/bash

# Sweep over submap sizes and power-of-2 overlap sizes, calling eval_tum.sh for each combination.
# Overlap sizes are powers of 2 (1, 2, 4, 8, ...) strictly less than submap_size / 2.

script_dir="$(cd "$(dirname "$0")" && pwd)"

submap_sizes=(8 16 32)

for submap_size in "${submap_sizes[@]}"; do
    max_overlap=$((submap_size / 2))
    overlap=1

    while [ "$overlap" -lt "$max_overlap" ]; do
        echo "========================================"
        echo "  submap_size=$submap_size  overlap=$overlap"
        echo "========================================"
        bash "$script_dir/eval_tum.sh" "$submap_size" "$overlap"
        overlap=$((overlap * 2))
    done
done

echo "Sweep complete."
