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


def accuracy_reward(completions, answer, choices, **kwargs):
    completion_content = [completion[0]["content"] for completion in completions] # 模型回复

    rewards = []
    for content, correct_option, choice in zip(completion_content, answer, choices):
        content_match = re.search(r'<answer>(.*?)</answer>', content)

        # 从模型的回复中提取模型的答案
        model_answer = content_match.group(1).strip().lower() if content_match else content.strip() # 提取模型回复中的答案
        model_option = None # 模型给出的选项
        for idx, choice_text in enumerate(choice):
            if idx == correct_option:
                if str(idx) in model_answer or choice_text.lower() in model_answer: # 如果模型回复中包含正确答案，reward = 1，继续检查
                    model_option = idx
            else:
                if str(idx) in model_answer or choice_text.lower() in model_answer: # 如果模型回复中包含错误答案，reward = 0， break
                    model_option = None
                    break

        if model_option == correct_option:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"completion: {content}\n")
                if model_option is None:
                    f.write(f"model_answer: {model_answer}\n")
                else:
                    f.write(f"model_answer: {model_option}: {choice[model_option]}\n")
                f.write(f"ground_truth: {correct_option}: {choice[correct_option]}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry two parts
reward_funcs_registry = {
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
    user_prompt += "Requirement: This is a multiple-choice question. Please select one answer choice based on the question and the provided options.\n"
    user_prompt += f"Hint: {hint}\n"
    return user_prompt

def make_scienceqa_user_prompt(question, choices, hint):
    user_prompt = f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        user_prompt += f"{idx}: {choice}\n"
    user_prompt += "Requirement: This is a multiple-choice question. Please select one answer choice based on the question and the provided options.\n"
    user_prompt += f"Hint: {hint}\n"
    return user_prompt

def main(script_args, training_args, model_args):
    # Get reward functions
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
