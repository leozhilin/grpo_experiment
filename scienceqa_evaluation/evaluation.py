import math
import re
import numpy as np
import torch
import multiprocessing as mp
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
from multiprocessing import Pool
import functools

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Initialize model and processor
def initialize_model(model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="models/qwen2_vl_2b"
    ).eval()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    return model, processor

# Create user prompt from question data
def make_scienceqa_user_prompt(question, choices, hint):
    user_prompt = f"Question: {question}\n" + \
                  "".join(f"{idx}: {choice}\n" for idx, choice in enumerate(choices)) + \
                  "Requirement: This is a multiple-choice question. Please select one answer choice based on the question and the provided options.\n" + \
                  f"Hint: {hint}\n"
    return user_prompt

# Filter dataset examples with missing images
def filter_vqa_items(example):
    return example["image"] is not None and example["image"] != ""

# Perform inference and evaluate model performance
def evaluate_model(rank, world_size, model_path):
    model, processor = initialize_model(model_path)

    dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir="datasets/science_qa").filter(filter_vqa_items)
    
    # Calculate split length
    split_length = math.ceil(len(dataset) / world_size)
    
    # Calculate start and end index for the current rank
    start_idx = int(rank * split_length)
    end_idx = int((rank + 1) * split_length)
    
    # Select dataset subset
    dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    acc_cnt, total_cnt = 0, 0
    system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."

    for example in dataset:
        choices, correct_option = example["choices"], example["answer"]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": system_prompt + "\n" + make_scienceqa_user_prompt(example["question"], choices, example["hint"])},
            ],
        }]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate model output
        generated_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        completion = processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        # Extract model's answer
        content_match = re.search(r'<answer>(.*?)</answer>', completion)
        model_answer = content_match.group(1).strip().lower() if content_match else completion.strip()

        # Determine model's choice
        model_option = next((idx for idx, choice_text in enumerate(choices) if str(idx) in model_answer or choice_text.lower() in model_answer), None)

        if model_option == correct_option:
            acc_cnt += 1
        total_cnt += 1

        print(f"model_answer: {model_answer}")
        print(f"ground_truth: {choices[correct_option]}")
        print(f"result: {model_option == correct_option}\n")

    acc_rate = acc_cnt / total_cnt if total_cnt > 0 else 0
    return acc_rate

def plot_results(model_paths, results):
    plt.figure(figsize=(10, 6))
    plt.plot(model_paths, results, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model Checkpoints')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('model_performance.png')


def main():
    model_paths = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-100",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-200",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-300",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-400",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-500",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-600",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-700",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-778",
    ]

    results = []
    for model_path in model_paths:
        n_gpus = torch.cuda.device_count() >= 2
        if n_gpus:
            logger.info('started generation')
            n_gpus = torch.cuda.device_count()
            with Pool(n_gpus) as pool:
                func = functools.partial(evaluate_model, world_size=n_gpus, model_path=model_path)
                acc_rates = pool.map(func, range(n_gpus))

            print("result list:\n", acc_rates)
            avg_acc_rate = np.mean(acc_rates)

            results.append(avg_acc_rate)
            print(f"Model Path: {model_path}, Accuracy: {avg_acc_rate}")
        else:
            avg_acc_rate = evaluate_model(0, 1, model_path)

    plot_results(model_paths, results)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()