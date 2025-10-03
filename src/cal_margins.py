
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import argparse
from typing import  Dict, List, Tuple
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed

def apply_chat_template(example: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, str]:
    """Applies the chat template to a single example."""
    prompt_messages = [{"role": "user", "content": example['prompt']}]
    chosen_messages = prompt_messages + [{"role": "assistant", "content": example['chosen']}]
    rejected_messages = prompt_messages + [{"role": "assistant", "content": example['rejected']}]

    formatted_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    full_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    prompt_len_char = len(formatted_prompt)
    chosen_only = full_chosen[prompt_len_char:]
    rejected_only = full_rejected[prompt_len_char:]

    return {
        'formatted_prompt': formatted_prompt,
        'chosen_response_only': chosen_only,
        'rejected_response_only': rejected_only,
    }

class MultiGPULogProbCalculator(nn.Module):
    """Wrapper for calculating log probabilities on multiple GPUs"""
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    @torch.no_grad()        
    def forward(self, batch_input_ids, batch_attention_mask, prompt_lengths, response_lengths):
        outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        batch_size = batch_input_ids.size(0)
        device = batch_input_ids.device
        sequence_logprobs = []
        
        for i in range(batch_size):
            p_len = prompt_lengths[i]
            r_len = response_lengths[i]
            seq_len = batch_input_ids.shape[1]

            logit_indices = range(p_len - 1, min(p_len + r_len - 1, seq_len - 1))
            token_indices = range(p_len, min(p_len + r_len, seq_len))

            if not logit_indices or not token_indices:
                sequence_logprobs.append(torch.tensor(0.0, device=device))
                continue

            response_logits = log_probs[i, logit_indices, :]
            response_token_ids = batch_input_ids[i, token_indices]
            gathered_logprobs = torch.gather(response_logits, -1, response_token_ids.unsqueeze(-1)).squeeze(-1)

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            non_padding_mask = (response_token_ids != pad_token_id)
            total_logprob = (gathered_logprobs * non_padding_mask).sum()
            sequence_logprobs.append(total_logprob)

        return torch.stack(sequence_logprobs)

def padding_to_max_length(input_ids, max_length, tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        padding_len = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_len
    return input_ids, attention_mask

class MultiGPUModelProcessor:
    def __init__(self, model_path: str, device_ids: List[int], tokenizer: AutoTokenizer, dtype=torch.bfloat16):
        self.device_ids = device_ids
        self.primary_device = f"cuda:{device_ids[0]}"
        self.tokenizer = tokenizer
        
        print(f"Loading model {model_path} on devices: {device_ids}")
        
        # Load model on primary device first
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.primary_device,
            trust_remote_code=True
        ).eval()
        
        # Wrap with DataParallel if multiple devices
        if len(device_ids) > 1:
            self.model = DataParallel(
                MultiGPULogProbCalculator(self.model, tokenizer),
                device_ids=device_ids
            )
        else:
            self.model = MultiGPULogProbCalculator(self.model, tokenizer)
    
    @torch.no_grad()
    def process_batch(self, batch_inputs: Dict, prompt_lengths: List[int], 
                     response_lengths: List[int], target_device: str = "cpu") -> torch.Tensor:
        """Process a batch and return results on target device"""
            # Move inputs to primary device
        batch_input_ids = batch_inputs['input_ids'].to(self.primary_device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.primary_device)
        
        # Convert lengths to tensors on the same device (for DataParallel)
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=self.primary_device)
        response_lengths_tensor = torch.tensor(response_lengths, device=self.primary_device)
        
        if isinstance(self.model, DataParallel):
            # DataParallel expects all inputs to be tensors
            logprobs = self.model(batch_input_ids, batch_attention_mask, 
                                prompt_lengths_tensor, response_lengths_tensor)
        else:
            # Single GPU case
            logprobs = self.model(batch_input_ids, batch_attention_mask, 
                                prompt_lengths, response_lengths)
        
        # Move results to target device
        return logprobs.to(target_device)
            

def setup_gpu_allocation(num_gpus: int) -> Tuple[List[int], List[int]]:
    """Allocate GPUs for policy and reference models"""
    if num_gpus < 2:
        raise ValueError("Need at least 2 GPUs for multi-GPU processing")
    
    # Split GPUs evenly between two models
    policy_gpus = list(range(0, num_gpus // 2))
    ref_gpus = list(range(num_gpus // 2, num_gpus))
    
    print(f"Policy model GPUs: {policy_gpus}")
    print(f"Reference model GPUs: {ref_gpus}")
    
    return policy_gpus, ref_gpus

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', type=str)
    parser.add_argument('--ref_model', type=str)
    parser.add_argument('--data_path', type=str, default='data/selectivedpo_data.jsonl')
    parser.add_argument('--output_path', type=str, default='/data/margin_results/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use (auto-detect if None)')
    parser.add_argument('--computation_device', type=str, default='cpu', help='Device for final computation')
    
    args = parser.parse_args()
    
    if not args.output_path:
        raise ValueError("Output path must be specified.")
    if not args.data_path:
        raise ValueError("Data path must be specified.")
    
    # Auto-detect GPU count if not specified
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    
    if args.num_gpus % 2 != 0:
        print(f"Warning: {args.num_gpus} GPUs detected. Using {args.num_gpus - 1} GPUs for even split.")
        args.num_gpus = args.num_gpus - 1
    
    return args

def main():
    args = get_args()
    dtype = torch.bfloat16
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    data_name = args.data_path.split('/')[-1].split('.')[0]
    model_name = args.policy_model.split('/')[-1]
    output_path = os.path.join(args.output_path, f"{model_name}_margins_{data_name}.jsonl")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping.")
        return

    if args.num_gpus < 2:
        raise ValueError("Multi-GPU mode requires at least 2 GPUs")
    
    print(f"Using {args.num_gpus} GPUs for parallel processing")
    
    # Setup GPU allocation
    policy_gpus, ref_gpus = setup_gpu_allocation(args.num_gpus)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model processors
    print("Initializing model processors...")
    policy_processor = MultiGPUModelProcessor(args.policy_model, policy_gpus, tokenizer, dtype)
    ref_processor = MultiGPUModelProcessor(args.ref_model, ref_gpus, tokenizer, dtype)

    # Load dataset
    datasets = pd.read_json(args.data_path, lines=True)
    datasets = datasets.dropna(subset=['prompt', 'chosen', 'rejected'])
    print(f"Loaded {len(datasets)} examples from {args.data_path}")

    results = []

    # Process data in batches
    for i in tqdm(range(0, len(datasets), args.batch_size), desc="Processing batches"):
        batch_df = datasets.iloc[i:i + args.batch_size]

        batch_inputs = {'input_ids': [], 'attention_mask': []}
        batch_prompt_lengths = []
        batch_chosen_response_lengths = []
        batch_rejected_response_lengths = []
        batch_original_data = []

        # Prepare batch data
        for idx, row in batch_df.iterrows():
            formatted = apply_chat_template(row.to_dict(), tokenizer)

            prompt_tokens = tokenizer(formatted['formatted_prompt'], add_special_tokens=False)
            chosen_tokens = tokenizer(formatted['chosen_response_only'] + tokenizer.eos_token, add_special_tokens=False)
            rejected_tokens = tokenizer(formatted['rejected_response_only'] + tokenizer.eos_token, add_special_tokens=False)

            prompt_len = len(prompt_tokens['input_ids'])
            chosen_len = len(chosen_tokens['input_ids'])
            rejected_len = len(rejected_tokens['input_ids'])

            (chosen_input_ids, chosen_attention_mask) = padding_to_max_length(
                prompt_tokens['input_ids'] + chosen_tokens['input_ids'], args.max_length, tokenizer)
            (rejected_input_ids, rejected_attention_mask) = padding_to_max_length(
                prompt_tokens['input_ids'] + rejected_tokens['input_ids'], args.max_length, tokenizer)

            batch_inputs['input_ids'].extend([chosen_input_ids, rejected_input_ids])
            batch_inputs['attention_mask'].extend([chosen_attention_mask, rejected_attention_mask])
            batch_prompt_lengths.extend([prompt_len, prompt_len])
            batch_chosen_response_lengths.append(chosen_len)
            batch_rejected_response_lengths.append(rejected_len)
            batch_original_data.append(row.to_dict())


        # Convert to tensors
        batch_tensor_inputs = {
            'input_ids': torch.tensor(batch_inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(batch_inputs['attention_mask'], dtype=torch.long)
        }

        all_response_lengths = []
        for i in range(len(batch_chosen_response_lengths)):
            all_response_lengths.extend([batch_chosen_response_lengths[i], batch_rejected_response_lengths[i]])

        # Process with both models in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            policy_future = executor.submit(
                policy_processor.process_batch,
                batch_tensor_inputs, batch_prompt_lengths, all_response_lengths, args.computation_device
            )
            
            ref_future = executor.submit(
                ref_processor.process_batch,
                batch_tensor_inputs, batch_prompt_lengths, all_response_lengths, args.computation_device
            )
            
            # Get results
            policy_logprobs = policy_future.result()
            ref_logprobs = ref_future.result()

        if policy_logprobs is None or ref_logprobs is None:
            print("Skipping batch due to processing error")
            continue

        # Ensure same device
        if policy_logprobs.device != ref_logprobs.device:
            policy_logprobs = policy_logprobs.to(args.computation_device)
            ref_logprobs = ref_logprobs.to(args.computation_device)

        # Separate chosen and rejected logprobs
        policy_chosen_logprobs = policy_logprobs[0::2]
        policy_rejected_logprobs = policy_logprobs[1::2]
        ref_chosen_logprobs = ref_logprobs[0::2]
        ref_rejected_logprobs = ref_logprobs[1::2]

        # Calculate margins and save results
        for j in range(len(policy_chosen_logprobs)):
            margin = policy_chosen_logprobs[j] - policy_rejected_logprobs[j] - (ref_chosen_logprobs[j] - ref_rejected_logprobs[j])
            
            result = {
                "prompt": batch_original_data[j]['prompt'],
                "chosen": batch_original_data[j]['chosen'],
                "rejected": batch_original_data[j]['rejected'],
                "policy_chosen_logprob": policy_chosen_logprobs[j].item(),
                "policy_rejected_logprob": policy_rejected_logprobs[j].item(),
                "ref_chosen_logprob": ref_chosen_logprobs[j].item(),
                "ref_rejected_logprob": ref_rejected_logprobs[j].item(),
                "margin": margin.item(),
            }
            results.append(result)


    # Save results
    with open(output_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"Successfully saved {len(results)} results to {output_path}")

if __name__ == "__main__":
    main()
