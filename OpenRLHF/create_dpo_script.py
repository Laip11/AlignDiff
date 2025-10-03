import argparse
import os

def create_dpo_script(
    model_path: str,
    dataset_path: str,
    sh_path: str = None,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_epochs: int = 1,
    train_batch_size: int = 128,
    micro_train_batch_size: int = 1,
    #shuffle_train_dataset: bool = False  
):

    model_name = os.path.basename(os.path.normpath(model_path))
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    save_path = f'saves/{model_name}/{model_name}_{dataset_name}'
    os.makedirs(save_path, exist_ok=True)

    if sh_path is None:
        script_dir = 'OpenRLHF/training_scripts'

        dataset_filename_safe = dataset_name.replace("/", "_")
        sh_path = f'{script_dir}/train_{model_name}_{dataset_filename_safe}.sh'
    
    #shuffle_flag = "--shuffle_train_dataset \\" if shuffle_train_dataset else ""

    # Construct the shell script content using an f-string
    script_content = f"""#!/bin/bash

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \\
   --save_path {save_path} \\
   --save_steps -1 \\
   --logging_steps 1 \\
   --eval_steps -1 \\
   --lr_warmup_ratio 0.1 \\
   --train_batch_size {train_batch_size} \\
   --micro_train_batch_size {micro_train_batch_size} \\
   --pretrain {model_path} \\
   --bf16 \\
   --max_epochs {max_epochs} \\
   --max_len 2048 \\
   --zero_stage 3 \\
   --learning_rate {learning_rate} \\
   --beta {beta} \\
   --dataset {dataset_path} \\
   --dataset_split train \\
   --apply_chat_template \\
   --chosen_key chosen \\
   --rejected_key rejected \\
   --flash_attn \\
   --load_checkpoint \\
   --gradient_checkpointing \\
   --seed 42 \\
   --wandb_run_name {save_path.split('/')[-1]} \\
EOF

if [[ "${{1}}" != "slurm" ]]; then
    deepspeed --module $training_commands
else
    echo "$training_commands"
fi
"""
    script_content = os.linesep.join([s for s in script_content.splitlines() if s.strip()])


    if sh_path:
        os.makedirs(os.path.dirname(sh_path), exist_ok=True)

    with open(sh_path, 'w') as f:
        f.write(script_content)

    print(f"✔ the training checkpoints will save at: {save_path}")
    print(f"✔ the executable shell script has been generated: {sh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define command-line arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained SFT model used for DPO training.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Name or path of the preference dataset.")
    parser.add_argument("--sh_path", type=str, default=None, help="[Optional] Path to save the generated .sh file.")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global batch size.")
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="Per-device batch size.")
    # parser.add_argument("--shuffle_train_dataset", action='store_true', help="If provided, the training dataset will be shuffled.")
    args = parser.parse_args()

    create_dpo_script(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        sh_path=args.sh_path,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        micro_train_batch_size=args.micro_train_batch_size,
        # shuffle_train_dataset=args.shuffle_train_dataset 
    )
