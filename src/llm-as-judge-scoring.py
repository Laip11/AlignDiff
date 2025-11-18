
import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm,trange
import pandas as pd
import datasets
import numpy as np
import argparse
import torch

template = '''You are an evaluator tasked with assigning a score from 1 to 9 based on the criteria below, to assess the quality of a specific aspect of the data entry,You do not need to provide an explanation.
### Evaluation Task:
Assess the given data entry based on the following aspect: **{evaluation_aspect}**. For this task, your goal is to evaluate how well the **{evaluation_aspect}** is demonstrated in the provided **Instruction** and **Response**. Be sure to consider the following:
- **{evaluation_aspect}** refers to a specific quality or characteristic in the data entry. 
- Consider both the **Instruction** (the directive given) and the **Response** (the output provided in answer to the instruction).
- The evaluation should be based on how clearly, accurately, or relevantly the task is fulfilled according to the criteria listed below.
- Only assess the given  **Instruction** and **Response**; do not consider any additional context or external information.

### Definition of {evaluation_aspect}:
- **{evaluation_aspect}** is {Definition}

### Scoring Criteria:
- **Score 1**: {criteria_for_1}  
- **Score 2**: {criteria_for_2}  
- **Score 3**: {criteria_for_3}  
- **Score 4**: {criteria_for_4}  
- **Score 5**: {criteria_for_5}  
- **Score 6**: {criteria_for_6}  
- **Score 7**: {criteria_for_7}  
- **Score 8**: {criteria_for_8}  
- **Score 9**: {criteria_for_9}  

Each score corresponds to a specific level of quality, from 1 (lowest) to 9 (highest). Use the provided criteria to select the appropriate score.

### Data Entry to Evaluate:
- **Instruction**: 
{instruction}

- **Response**: 
{output}

### Response Format:
- Output a single score between 1 and 9 in the format “score: <the score you give>” based on your evaluation of the Instruction and Response. Do not provide any additional explanations or notes.'''

evaluation_criteria = {
    ### dpo
    "helpfulness": {
        "definition": "Assess whether the response provides valuable and actionable information that directly aids the user in achieving their objective.",
        "scoring_standards": {
            1: "Response is entirely unhelpful and irrelevant, offering no value.",
            2: "Response is mostly unhelpful, with very little relevance to the user’s needs.",
            3: "Provides limited help, but the response is vague, incomplete, or only partially relevant.",
            4: "Response is somewhat helpful, but important details or steps are missing.",
            5: "Moderately helpful, addressing the main query but lacking depth or precision.",
            6: "Mostly helpful and clear, with minor gaps or room for further detail.",
            7: "Helpful and mostly comprehensive, fulfilling the majority of the user's request.",
            8: "Highly helpful and clear, with only minimal areas for potential improvement.",
            9: "Exceptionally helpful, fully addressing the user’s needs in a complete and actionable manner."
        }
    },
    "honesty": {
        "definition": "Evaluate whether the response is truthful and avoids misleading or fabricating information.",
        "scoring_standards": {
            1: "Response is entirely dishonest, fabricated, or intentionally misleading.",
            2: "Response includes significant falsehoods or misleading information.",
            3: "Some information is correct, but key elements are false or deceptive.",
            4: "Mostly truthful, but contains noticeable inaccuracies or vague claims.",
            5: "Generally honest, though minor factual errors or omissions exist.",
            6: "Honest and clear, with very few small inaccuracies or overly cautious phrasing.",
            7: "Mostly accurate and honest, with no significant misleading elements.",
            8: "Entirely honest and accurate, only minor potential for clarification needed.",
            9: "Flawlessly honest, all information is accurate and presented transparently."
        }
    },
    "instruction_following": {
        "definition": "Determine if the response adheres precisely to the instructions provided, fulfilling the request as intended.",
        "scoring_standards": {
            1: "Completely disregards the instruction, with no relation to the request.",
            2: "Fails to follow the instruction significantly, with only minor relevant elements.",
            3: "Partially follows the instruction, but with major omissions or errors.",
            4: "Somewhat follows the instruction, though important aspects are overlooked.",
            5: "Follows the instruction moderately well, with noticeable gaps or misinterpretations.",
            6: "Mostly adheres to the instruction, with minor deviations or missed nuances.",
            7: "Adheres to the instruction well, with only slight areas for improvement.",
            8: "Accurately follows the instruction, with minimal need for refinement.",
            9: "Perfectly adheres to the instruction, fulfilling every aspect flawlessly."
        }
    },
    "truthfulness": {
        "definition": "Assess whether the response is factually accurate and based on verified knowledge or reasoning.",
        "scoring_standards": {
            1: "Response is completely false, with no factual basis or accuracy.",
            2: "Response is mostly false, with very few correct facts.",
            3: "Response has a mix of true and false information, with major inaccuracies.",
            4: "Response is somewhat accurate but includes noticeable factual errors.",
            5: "Response is generally accurate but contains some minor factual inaccuracies.",
            6: "Mostly truthful and fact-based, with minimal errors or ambiguities.",
            7: "Highly accurate and truthful, with no significant errors or misleading information.",
            8: "Entirely truthful and accurate, with only trivial areas for clarification or nuance.",
            9: "Flawlessly truthful, presenting facts with utmost accuracy and precision."
        }
    }
}
def add_criteria(template, aspect,instruction, output):
    criteria = evaluation_criteria[aspect]
    return template.format(evaluation_aspect=aspect, 
                           Definition=criteria['definition'],
                           criteria_for_1=criteria['scoring_standards'][1],
                           criteria_for_2=criteria['scoring_standards'][2],
                           criteria_for_3=criteria['scoring_standards'][3],
                           criteria_for_4=criteria['scoring_standards'][4],
                           criteria_for_5=criteria['scoring_standards'][5],
                           criteria_for_6=criteria['scoring_standards'][6],
                           criteria_for_7=criteria['scoring_standards'][7],
                           criteria_for_8=criteria['scoring_standards'][8],
                           criteria_for_9=criteria['scoring_standards'][9],
                           instruction=instruction, output=output)


def calculate_expected_score(logprob_data):
    # 初始化为负无穷的 numpy 数组，表示1-5分的logprob
    score_map = np.full(9, -np.inf)
    # 遍历 logprob_data
    for key in logprob_data.keys():
        token = logprob_data[key].decoded_token
        if token in ['1', '2', '3', '4', '5','6','7','8','9']:
            score_map[int(token) - 1] = logprob_data[key].logprob  # 按索引存储 logprob
        else:
            continue
    if np.all(score_map == -np.inf):
        return 0, 0
    
    max_logit_index = np.argmax(score_map)  
    max_logit_score = max_logit_index + 1  


    exp_logprobs = np.exp(score_map)  
    probabilities = exp_logprobs / np.sum(exp_logprobs)

    expectation = np.dot(probabilities, np.arange(1, 10))  
    return expectation, max_logit_score


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--data_dir", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--output_dir", type=str, default='./data/')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--aspect", type=str, default="helpfulness")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    scores_ids_ls = tokenizer.convert_tokens_to_ids([str(i) for i in range(1,10)])
    model_name = args.model_name.split('/')[-1]

    llm = LLM(model = args.model_name,
              tokenizer= args.model_name,
              dtype=torch.bfloat16,
              tensor_parallel_size=torch.cuda.device_count(),
              max_model_len = 10000
              )
    
    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0,
        logprobs=20,  # 返回每个 token 的概率
        stop_token_ids=scores_ids_ls,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_file = os.path.join(args.output_dir, f"{model_name}_{args.aspect}.jsonl")
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        exit(0)


    print('Loading data and processing the prompt...')
    all_data = datasets.load_dataset(args.data_dir,split='train_prefs',cache_dir='./data/')

    prompt_chosen_ls = [ 
        add_criteria(
            template=template, 
            aspect=args.aspect,
            instruction = data['prompt'],
            output = data['chosen'][1]['content']
            ) 
            for data in tqdm(all_data,desc='Processing chosen data')]
    prompt_rejected_ls = [
        add_criteria(
            template=template, 
            aspect=args.aspect,
            instruction = data['prompt'],
            output = data['rejected'][1]['content']
            ) 
            for data in tqdm(all_data,desc = 'Processing rejected data')]
    all_prompt = pd.DataFrame({'prompt_chosen':prompt_chosen_ls,'prompt_rejected':prompt_rejected_ls})
    for prompt_type in tqdm(all_prompt.columns):
        all_score = []
        for idx in trange(0, len(all_prompt), args.batch_size):

            batch_messages = [{"role": "user", "content": prompt} for prompt in all_prompt[prompt_type][idx:idx+args.batch_size]]
            text_batch = [tokenizer.apply_chat_template([message],tokenize=False,add_generation_prompt = True) for message in batch_messages]

            outputs = llm.generate(text_batch, sampling_params,use_tqdm=False)

            for output in outputs:
                generated_text = output.outputs[0].text
                logprobs_ls = output.outputs[0].logprobs  # 获取每个 token 的概率
                expectation, max_logit_score = calculate_expected_score(logprobs_ls[-1])
                
                all_score.append(expectation)
        all_prompt[f'{prompt_type}_{args.aspect}_score'] = all_score
    
    all_prompt.to_json(output_file, orient='records', lines=True)
        
            
            

    