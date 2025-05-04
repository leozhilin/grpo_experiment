# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

from math import isnan

def check_nan_value(bbox_list: list[dict[str, float]]):
    for bbox in bbox_list:
        if isnan(bbox["Confidence"]):
            has_nan = True
            return has_nan
    return False


def replace_nan_confidence_value(
        bbox_list: list[dict[str, float]], value: float = 0.):
    for bbox in bbox_list:
        if isnan(bbox["Confidence"]):
            bbox["Confidence"] = value
    return bbox_list

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

# def extract_bbox(response):
#     start_tag = "<answer>"
#     end_tag = "</answer>"
#     input_str = response
#     # Check if the start tag is in the string
#     if start_tag in input_str:
#         # Extract the content between the start tag and end tag
#         start_idx = input_str.find(start_tag) + len(start_tag)
#         end_idx = input_str.find(end_tag)
        
#         # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
#         if end_idx == -1:
#             end_idx = len(input_str)
    
#         content_str = input_str[start_idx:end_idx]
    
#         # Check if it ends with a closing bracket, if not, fix it
#         if not content_str.endswith("]"):
#             # If the string is truncated, remove the incomplete part
#             content_str = content_str.rsplit("},", 1)[0] + "}]"
    
#         # Replace single quotes with double quotes for valid JSON
#         content_str_corrected = content_str.replace("'", '"')
    
#         # Convert the corrected string to a list of dictionaries (JSON format)
#         try:
#             bbox_list = json.loads(content_str_corrected)
#         except json.JSONDecodeError as e:
#             bbox_list = None
#     else:
#         bbox_list = None
#     return bbox_list

# def calculate_iou(bbox1, bbox2):
#     x1, y1, x2, y2 = bbox1
#     x1_2, y1_2, x2_2, y2_2 = bbox2

#     xi1 = max(x1, x1_2)
#     yi1 = max(y1, y1_2)
#     xi2 = min(x2, x2_2)
#     yi2 = min(y2, y2_2)
    
#     if xi2 <= xi1 or yi2 <= yi1:
#         return 0.0
    
#     intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
#     area1 = (x2 - x1) * (y2 - y1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

#     union_area = area1 + area2 - intersection_area
    
#     iou = intersection_area / union_area
#     return iou

# def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
#     list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
#     iou_results = []
    
#     matched_list1_indices = set()

#     for bbox2 in list2_sorted:
#         max_iou = 0
#         matched_bbox1 = -1
#         best_iou = 0
#         for i, bbox1 in enumerate(list1):
#             if i not in matched_list1_indices:
#                 iou = calculate_iou(bbox1['Position'], bbox2['Position'])
#                 if iou > best_iou:
#                     best_iou = iou
#                     matched_bbox1 = i

#         if best_iou > iou_threshold:
#             iou_results.append((best_iou, bbox2['Confidence']))
#             matched_list1_indices.add(matched_bbox1)
#         else:
#             iou_results.append((0, bbox2['Confidence']))
    
#     ### [(0.7192676547515258, 1.0), (0, 0.7)]
#     return iou_results

# def remove_duplicates(bbox_list):
#     seen = set()
#     unique_bboxes = []
    
#     for bbox in bbox_list:
#         # Convert the position tuple to a tuple for set hashing
#         position_tuple = tuple(bbox['Position'])
        
#         if position_tuple not in seen:
#             seen.add(position_tuple)
#             unique_bboxes.append(bbox)
    
#     return unique_bboxes

# # V1
# def compute_reward_iou(iou_results):
#     iou_reward = 0.0
#     confidence_reward = 0.0
#     for i in range(len(iou_results)):
#         temp_iou = iou_results[i][0]
#         temp_confidence = iou_results[i][1]

#         temp_iou_reward = temp_iou
#         if temp_iou == 0:
#             temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
#         else:
#             temp_confidence_reward = temp_confidence

#         iou_reward += temp_iou_reward
#         confidence_reward += temp_confidence_reward
        
#     iou_reward = iou_reward/len(iou_results)
#     confidence_reward = confidence_reward/len(iou_results)
#     return iou_reward

# # V2
# def compute_reward_iou_v2(iou_results, len_gt):
#     iou_reward = 0.0
#     confidence_reward = 0.0
#     for i in range(len(iou_results)):
#         temp_iou = iou_results[i][0]
#         temp_confidence = iou_results[i][1]

#         temp_iou_reward = temp_iou
#         if temp_iou == 0:
#             temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
#         else:
#             temp_confidence_reward = temp_confidence

#         iou_reward += temp_iou_reward
#         confidence_reward += temp_confidence_reward
        
#     if len_gt>=len(iou_results):
#         iou_reward = iou_reward/len_gt
#     else:
#         iou_reward = iou_reward/len(iou_results)
#     return iou_reward

# def compute_reward_confidence(iou_results):
#     iou_reward = 0.0
#     confidence_reward = 0.0
#     for i in range(len(iou_results)):
#         temp_iou = iou_results[i][0]
#         temp_confidence = iou_results[i][1]

#         temp_iou_reward = temp_iou
#         if temp_iou == 0:
#             temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
#         else:
#             temp_confidence_reward = temp_confidence

#         iou_reward += temp_iou_reward
#         confidence_reward += temp_confidence_reward
        
#     iou_reward = iou_reward/len(iou_results)
#     confidence_reward = confidence_reward/len(iou_results)
#     return confidence_reward

# def accuracy_reward_iou(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         student_answer_bbox = []
#         ground_truth_bbox = []
#         iou_results = []
#         show_flage = 0

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 show_flage = 1
#                 # Extract answer from solution if it has think/answer tags
#                 ground_truth = sol.strip()
#                 # Extract answer from content if it has think/answer tags
#                 content_match = re.search(r'<answer>(.*?)</answer>', content)
#                 student_answer = content_match.group(1).strip() if content_match else content.strip()
#                 student_answer = '<answer>'+student_answer+'</answer>'

#                 # fix format error
#                 student_answer = student_answer.replace("[[",'[')  
#                 student_answer = student_answer.replace("]]",']')  
#                 student_answer = student_answer.replace("\n",'')  
#                 # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
#                 ground_truth_bbox = extract_bbox(ground_truth)
#                 student_answer_bbox = extract_bbox(student_answer)
#                 # pdb.set_trace()
#                 if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
#                     reward = 0.0
#                 else:
#                     student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
#                     iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
#                     ### new iou reward
#                     reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
#                     if reward>1:
#                         reward = 1.0
#                 # reward = reward * 0.1
#             except Exception:
#                 pass  # Keep reward as 0.0 if both methods fail
                
#         rewards.append(reward)
#         # import pdb; pdb.set_trace()
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
#                 f.write(f"content: {content}\n")
#                 f.write(f"sol: {sol}\n")
#                 if show_flage==1:
#                     f.write(f"student_answer_bbox: {student_answer_bbox}\n")
#                     f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
#                     if student_answer_bbox!=None:
#                         f.write(f"iou_results: {iou_results}\n")
#         show_flage = 0 
#     return rewards

# def accuracy_reward_confidence(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         student_answer_bbox = []
#         ground_truth_bbox = []
#         iou_results = []
#         show_flage = 0

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 show_flage = 1
#                 # Extract answer from solution if it has think/answer tags
#                 ground_truth = sol.strip()
#                 # Extract answer from content if it has think/answer tags
#                 content_match = re.search(r'<answer>(.*?)</answer>', content)
#                 student_answer = content_match.group(1).strip() if content_match else content.strip()
#                 student_answer = '<answer>'+student_answer+'</answer>'

#                 # fix format error
#                 student_answer = student_answer.replace("[[",'[')
#                 student_answer = student_answer.replace("]]",']')
#                 student_answer = student_answer.replace("\n",'')
#                 # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
#                 ground_truth_bbox = extract_bbox(ground_truth)
#                 student_answer_bbox = extract_bbox(student_answer)
#                 # pdb.set_trace()
#                 if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
#                     reward = 0.0
#                 else:
#                     student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates

#                     # fix nan bug
#                     if check_nan_value(student_answer_bbox):
#                         err_msg = (
#                             f"student answer: {student_answer}\n"
#                             f"ground truth: {ground_truth}\n"
#                             f"Replace the NaN with {0.0}\n"
#                         )
#                         print(err_msg)
#                         student_answer_bbox = (
#                             replace_nan_confidence_value(
#                                 student_answer_bbox, 0.0
#                             )
#                         )

#                     iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
#                     reward = compute_reward_confidence(iou_results)
#                     if reward>1:
#                         reward = 1.0
#                     if reward<0:
#                         reward = 0.0
#             except Exception:
#                 pass  # Keep reward as 0.0 if both methods fail
                
#         rewards.append(reward)
#         # import pdb; pdb.set_trace()
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
#                 f.write(f"content: {content}\n")
#                 f.write(f"sol: {sol}\n")
#                 if show_flage==1:
#                     f.write(f"student_answer_bbox: {student_answer_bbox}\n")
#                     f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
#                     if student_answer_bbox!=None:
#                         f.write(f"iou_results: {iou_results}\n")
#         show_flage = 0 
#     return rewards

def accuracy_reward(completions, answer, choices, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    answer_option = answer
    answer_text = [choice[int(option)] for choice, option in zip(choices, answer_option)]
    # print("~~~~~~~~~~~~~~~~~\n")
    rewards = []
    for content, option, text in zip(completion_contents, answer_option, answer_text):
        reward = 0.0
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip().lower() if content_match else content.strip()
        # print("student_answer", student_answer)
        if str(option) in student_answer or text.lower() in student_answer:
            reward = 1.0
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # print("~~~~~~~~~~~~~~~~~\n")
    # print(completion_contents)
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry three parts
reward_funcs_registry = {
    # "accuracy_iou": accuracy_reward_iou,
    # "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
    "accuracy": accuracy_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_mmbench_user_prompt(question, hint, a, b, c, d):
    user_prompt = f"Question: {question}\n"
    if a is not None:
        user_prompt += f"A: {a}\n"
    if b is not None:
        user_prompt += f"B: {b}\n"
    if c is not None:
        user_prompt += f"C: {c}\n"
    if d is not None:
        user_prompt += f"D: {d}\n"
    user_prompt += "Requirement: This is a multiple-choice question. Please select the correct answer based on the question and the provided options.\n"
    user_prompt += f"Hint: {hint}\n"
    return user_prompt

def make_scienceqa_user_prompt(question, choices, hint):
    user_prompt = f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        user_prompt += f"{idx}: {choice}\n"
    user_prompt += "Requirement: This is a multiple-choice question. Please select the correct answer based on the question and the provided options.\n"
    user_prompt += f"Hint: {hint}\n"
    return user_prompt

def main(script_args, training_args, model_args):
    # Get reward functions
    # script_args.reward_funcs = ['accuracy_iou','accuracy_confidence','format']
    script_args.reward_funcs = ['format', 'accuracy']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface 
    # 从hf-mirror下载数据集
    if script_args.dataset_name == "datasets/mm_bench/MMBench_DEV_EN.tsv":
        print("datasets/mm_bench/MMBench_DEV_EN.tsv\n")
        dataset = load_dataset("csv", data_files=script_args.dataset_name, delimiter="\t")
    elif script_args.dataset_name == "derek-thomas/ScienceQA":
        dataset = load_dataset("derek-thomas/ScienceQA", cache_dir="datasets/science_qa")
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="datasets/math_vista")

    # Format into conversation
    # def make_conversation(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {"role": "user", "content": example["question"]},
    #         ],
    #     }

    def make_conversation_image(example):
        # example['image'] = example['decoded_image']
        return {
            "prompt": [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT + "\n" + make_scienceqa_user_prompt(example["question"], example["choices"], example["hint"])},
                    ],
                },
            ],
        }
    def filter_vqa_items(example):
        # Define your condition here. For example, remove items without an image
        return example["image"] is not None and example["image"] != ""
    
    # Apply the filter to the dataset
    dataset = dataset.filter(filter_vqa_items)
    dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.dataset_train_split = "train"
    print("training_args\n", training_args)
    print("script_args\n", script_args)
    main(script_args, training_args, model_args)
