#!/bin/bash

# 文件列表
file_list=(
   your_filted_data_path_created_by_get_filtered_data.py

)
for in_file in "${file_list[@]}"
do
    out_folder="${in_file%.jsonl}"

    mkdir -p "$out_folder"

    python src/dpo_transform_format.py --in_file "$in_file" --out_folder "$out_folder"
done