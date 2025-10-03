from transformers import AutoTokenizer
import json
import pandas as pd
import numpy as np

def get_len_gap(data_path):

    data = get_data(data_path)
    if type(data.iloc[0]['chosen']) is not str:
        data['chosen'] = data['chosen'].apply(lambda x: x[1]['content'])
        data['rejected'] = data['rejected'].apply(lambda x: x[1]['content'])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data['chosen_len'] = data['chosen'].apply(lambda x: len(tokenizer.encode(x)))
    data['rejected_len'] = data['rejected'].apply(lambda x: len(tokenizer.encode(x)))
    data['len_gap'] = data['chosen_len'] - data['rejected_len']
    return data

def get_data(data_path):
    import json
    import pandas as pd
    from datasets import load_dataset
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            data = pd.DataFrame([json.loads(line) for line in f])
    elif data_path.endswith('.arrow'):
        data = load_dataset("arrow", data_files=data_path, split="train").to_pandas()
    else:
        raise ValueError('data_path must be json or arrow')
    return data

def add_logprobs_and_ppl(data1, data):

    import numpy as np
    import pandas as pd

    # add the original logprob
    data1['original_chosen_logprobs'] = data['ref_chosen_logprob']
    data1['original_rejected_logprobs'] = data['ref_rejected_logprob']

    # Calulate the average logprob
    avg_chosen = data['ref_chosen_logprob'] / data1['chosen_len']
    avg_rejected = data['ref_rejected_logprob'] / data1['rejected_len']
    data1['original_avg_chosen_logprobs'] = avg_chosen
    data1['original_avg_rejected_logprobs'] = avg_rejected

    # Calculate PPL
    data1['original_chosen_ppl'] = np.exp(-avg_chosen)
    data1['original_rejected_ppl'] = np.exp(-avg_rejected)

    # Calculate logprob gap
    data1['original_avg_logprobs_gap'] = avg_chosen - avg_rejected
    data1['original_ppl_gap'] = data1['original_chosen_ppl'] - data1['original_rejected_ppl']

    return data1[(data1['chosen_len'] > 0)&(data1['rejected_len'] > 0)]

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ad_data_path', type=str, default='data/Qwen2_5-7B-SFT_all_uf_margins.jsonl')
    parser.add_argument('--im_data_path', type=str, default='data/Qwen2_5-7B-SFT_all_uf_implict_margins.jsonl')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    ad_data_path = args.ad_data_path
    im_data_path = args.im_data_path

    data1 = get_len_gap(ad_data_path)
    data = get_len_gap(im_data_path)

    data1 = add_logprobs_and_ppl(data1, data)

    if 'qwen' in args.ad_data_path.lower():
        tau = 20
        model = 'Qwen2_5-7B-SFT'
    elif 'mistral' in args.ad_data_path.lower():
        tau = 80
        model = 'Mistral-7B-SFT'
    else:
        tau = 20
        model = 'LLaMA-3-8B-SFT'

    # filp the data according to the ad
    correct_data = data1[data1['margin'] > tau].copy()

    reverse_data = data1[data1['margin'] < -tau].copy()

    # fliping chosen and rejected
    reverse_data['chosen'], reverse_data['rejected'] = reverse_data['rejected'], reverse_data['chosen']

    # reverse_data['margin'] = -reverse_data['margin']
    # reverse_data['external_margin'] = -reverse_data['external_margin']

    # combining the data
    combined_data = pd.concat([correct_data, reverse_data], ignore_index=True)

    # filtering the data with high 
    combined_data.sort_values(by='original_avg_logprobs_gap', ascending=True, inplace=True)
    filtered_data = combined_data[['prompt', 'chosen', 'rejected']][:30000]
    filtered_data.to_json(f'data/{model}_original_high_original_avg_logprobs_gap_30k_data.jsonl', lines = True,orient='records')
