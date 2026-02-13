#!/bin/bash

abs_dir="/home/<user>/Documents"
submap_size=${1:-16} # Default to 16 if not provided
dataset_path="${abs_dir%/}/MASt3R-SLAM/datasets/tum/"
gt_path="${abs_dir%/}/MASt3R-SLAM/datasets/tum/"
log_path="$(pwd)/logs/tum_results_w${submap_size}.txt"

mkdir -p "$(pwd)/logs"

datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# Number of full runs
n=1  # <-- change as needed

# If file doesn't exist, write header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    count=0

    for dataset in "${datasets[@]}"; do
        echo "Running main.py on $dataset (Run $run)"
        dataset_name="${dataset_path}${dataset}/rgb"
        python main.py --image_folder "$dataset_name" --max_loops 1 --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 --submap_size "$submap_size" --log_results --skip_dense_log --log_path "$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt"
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        est_path="$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        echo "$run,$dataset,$rmse" >> "$log_path"

        total_rmse=$(echo "$total_rmse + $rmse" | bc -l)
        count=$((count + 1))
    done

    avg_rmse=$(echo "$total_rmse / $count" | bc -l)
    echo "$run,Average,$avg_rmse" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
done
