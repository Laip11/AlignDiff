#!/bin/bash

set -e 

# --- Configure your paths and parameters here ---
cd OpenRLHF
MODEL_PATHS=(
HuggingFaceH4/mistral-7b-sft-beta
)

DATASET_PATHS=(
your_data_path
)

LR=5e-7
BETA=0.01
EPOCHS=1
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=2


echo "Starting batch script generation..."
for model_path in "${MODEL_PATHS[@]}"; do
    for dataset_path in "${DATASET_PATHS[@]}"; do
        
        echo "------------------------------------------------"
        echo "Generating script for the following combination:"
        echo "  - Model: $model_path"
        echo "  - Dataset: $dataset_path"
        echo ""

        # Call the Python script and pass in the current model_path and dataset_path
        python create_dpo_script.py \
            --model_path "$model_path" \
            --dataset_path "$dataset_path" \
            --learning_rate "$LR" \
            --beta "$BETA" \
            --max_epochs "$EPOCHS" \
            --train_batch_size "$GLOBAL_BATCH_SIZE" \
            --micro_train_batch_size "$MICRO_BATCH_SIZE" 
            # --shuffle_train_dataset
    done
done

echo "------------------------------------------------"
echo "Script generation completed"