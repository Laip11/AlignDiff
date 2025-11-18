import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def select_em(df):

    # The 'em' data can be obtained from AlignDiff/src/llm-as-judge-scoring.py

    if 'em' not in df.columns:
        try:
            return df.nlargest(30000, 'margin')
        except KeyError:
            raise ValueError("DataFrame must contain either 'em' or 'margin' column.")
    else:
        return df.nlargest(30000, 'em')



def select_im(df):

    # The 'im' data can be obtained from AlignDiff/scripts/get_margins.sh

    if 'im' not in df.columns:
        try:
            return df.nlargest(30000, 'margin')
        except KeyError:
            raise ValueError("DataFrame must contain either 'im' or 'margin' column.")
    else:
        return df.nlargest(30000, 'im')


def select_im_and_em(df, em_col='em', im_col='im', top_k=30000, M1=-2, M2=30):

# The 'im' and 'em' columns can be obtained following the instructions above

    """
    Compute joint probability for each row and select top_k rows with highest joint probability.

    Parameters:
    - df (pd.DataFrame): input DataFrame containing external and implicit margins
    - em_col (str): column name for external margin
    - im_col (str): column name for implicit margin
    - top_k (int): number of top rows to select
    - M1 (float): lower bound for clipping
    - M2 (float): upper bound for clipping

    Returns:
    - pd.DataFrame: top_k rows with highest joint probability
    """
    def normalize(m):
        clipped = np.clip(m, M1, M2)
        return (clipped - M1) / (M2 - M1)
    
    p_ex = normalize(df[em_col].values)
    p_im = normalize(df[im_col].values)
    
    numerator = p_ex * p_im
    denominator = numerator + (1 - p_ex) * (1 - p_im)
    
    joint_prob = numerator / denominator
    
    # Add joint probability as a new column
    df = df.copy()
    df['joint_prob'] = joint_prob
    
    # Select top_k rows
    top_df = df.nlargest(top_k, 'joint_prob')
    
    return top_df




def select_rip(df,threshold=0.126):
    droped_df = df[df['em']>threshold]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    droped_df['rejected_len'] = droped_df['rejected'].apply(lambda x: len(tokenizer.encode(x)))

    return droped_df.nlargest(30000, 'rejected_len')


def select_LCPP(df):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    df['chosen_len'] = df['chosen'].apply(lambda x: len(tokenizer.encode(x)))

    return df.nlargest(30000, 'chosen_len')


def select_ppl_gap(df):
    ## you can get the data from  AlignDiff/scripts/get_margins.sh -> AlignDiff/scripts/get_filtered_data.sh
    return df.nlargest(30000, 'original_avg_logprobs_gap')


def get_SDPO_data():

    ## you can directly get the data from https://github.com/glorgao/SelectiveDPO/tree/main/selective-dpo/curricula
    ## then, you can get the bottom 50% data based on the 'learning order' column

    return None