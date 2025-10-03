#    your_policy_model_path (the postive policy model path)
ckpt_ls=(
   your_policy_model_path
   )

# the reference model path or inverse policy model path
ref_model=your_ref_model_path

for ckpt in "${ckpt_ls[@]}"; do
    echo "Running for checkpoint: $ckpt"
    CUDA_VISIBLE_DEVICES=0,1 python src/cal_margins.py \
        --policy_model "$ckpt" \
        --ref_model "$ref_model" \
        --batch_size 2 \
        --computation_device cuda:0
done