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
from prompt_utils import (
    system_prompt,
    make_question_prompt,
    make_critic_prompt,
    make_reflection_prompt
)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_model(reasoner_model_path, critic_model_path="Qwen/Qwen2-VL-2B-Instruct"):
    """
    初始化推理模型和评判模型
    
    Args:
        reasoner_model_path: 推理模型的路径
        critic_model_path: 评判模型的路径，默认为Qwen2-VL-2B-Instruct
    
    Returns:
        reasoner_model: 推理模型
        reasoner_processor: 推理模型的处理器
        critic_model: 评判模型
        critic_processor: 评判模型的处理器
    """
    # 初始化推理模型
    reasoner_model = Qwen2VLForConditionalGeneration.from_pretrained(
        reasoner_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="models/qwen2_vl_2b"
    ).eval()
    
    reasoner_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # 初始化评判模型
    critic_model = Qwen2VLForConditionalGeneration.from_pretrained(
        critic_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="models/qwen2_vl_2b"
    ).eval()
    
    critic_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    return reasoner_model, reasoner_processor, critic_model, critic_processor


def filter_vqa_items(example):
    """
    过滤掉没有图片的样本
    
    Args:
        example: 数据集样本
    
    Returns:
        bool: 是否保留该样本
    """
    return example["image"] is not None and example["image"] != ""


def generate_model_output(model, processor, messages, max_new_tokens=512):
    """
    生成模型输出
    
    Args:
        model: 模型
        processor: 处理器
        messages: 输入消息
        max_new_tokens: 最大生成token数
    
    Returns:
        模型生成的文本
    """
    # 准备输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    output = processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True
    )[0]
    
    return output


def evaluate_model(rank, world_size, reasoner_model_path, critic_model_path):
    """
    评估模型性能
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
        reasoner_model_path: 推理模型的路径
        critic_model_path: 评判模型的路径
    
    Returns:
        准确率
    """
    # 初始化模型
    reasoner_model, reasoner_processor, critic_model, critic_processor = initialize_model(reasoner_model_path, critic_model_path)

    # 加载数据集
    dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir="datasets/science_qa").filter(filter_vqa_items)
    
    # 计算数据分片
    split_length = math.ceil(len(dataset) / world_size)
    start_idx = int(rank * split_length)
    end_idx = int((rank + 1) * split_length)
    dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    # 初始化评估指标
    acc_cnt, total_cnt = 0, 0

    # 遍历数据集进行评估
    for example in dataset:
        choices, correct_option = example["choices"], example["answer"]
        
        # 第一步：初始推理
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": system_prompt + "\n" + make_question_prompt(example["question"], choices, example["hint"])},
            ],
        }]
        first_reasoning = generate_model_output(reasoner_model, reasoner_processor, messages)

        # 第二步：评判
        critic_prompt = make_critic_prompt(example["question"], choices, example["hint"], first_reasoning)
        critic_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": critic_prompt},
            ],
        }]
        critic_output = generate_model_output(critic_model, critic_processor, critic_messages)

        # 第三步：最终推理（带反思）
        reflection_prompt = make_reflection_prompt(example["question"], choices, example["hint"], critic_output)
        final_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": system_prompt + "\n" + reflection_prompt},
            ],
        }]
        final_reasoning = generate_model_output(reasoner_model, reasoner_processor, final_messages)

        # 提取答案并评估
        content_match = re.search(r'<answer>(.*?)</answer>', final_reasoning)
        model_answer = content_match.group(1).strip().lower() if content_match else final_reasoning.strip()
        model_option = next((idx for idx, choice_text in enumerate(choices) if str(idx) in model_answer or choice_text.lower() in model_answer), None)

        # 更新评估指标
        if model_option == correct_option:
            acc_cnt += 1
        total_cnt += 1

        # 打印详细信息
        print(f"First reasoning: {first_reasoning}")
        print(f"Critic output: {critic_output}")
        print(f"Final reasoning: {final_reasoning}")
        print(f"model_answer: {model_answer}")
        print(f"ground_truth: {correct_option} - {choices[correct_option]}")
        print(f"result: {model_option == correct_option}\n")

    # 计算准确率
    acc_rate = acc_cnt / total_cnt if total_cnt > 0 else 0
    return acc_rate


def plot_results(plt_title, model_paths, results, save_path):
    """
    绘制模型性能对比图
    
    Args:
        model_paths: 模型路径列表
        results: 对应的准确率列表
    """
    plt.figure(figsize=(10, 6))
    plt.plot(model_paths, results, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model Checkpoints')
    plt.ylabel('Accuracy')
    plt.title(plt_title)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)


def main():
    """
    主函数：加载模型并执行评估
    """
    # 定义要评估的模型路径
    reasoner_model_paths = [
        # "Qwen/Qwen2-VL-2B-Instruct",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-100",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-200",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-300",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-400",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-500",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-600",
        # "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-700",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-778",
    ]
    critic_model_paths = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "models/ckpt/qwen2_2b_scienceqa_ckpt/checkpoint-778",
    ]

    # 对每个critic模型进行评估
    for critic_model_path in critic_model_paths:
        # 评估每个reasoner模型
        results = []
        for reasoner_model_path in reasoner_model_paths:
            n_gpus = torch.cuda.device_count() >= 2
            if n_gpus:
                logger.info('started generation')
                n_gpus = torch.cuda.device_count()
                with Pool(n_gpus) as pool:
                    func = functools.partial(evaluate_model, world_size=n_gpus, reasoner_model_path=reasoner_model_path, critic_model_path=critic_model_path)
                    acc_rates = pool.map(func, range(n_gpus))

                print("result list:\n", acc_rates)
                avg_acc_rate = np.mean(acc_rates)

                results.append(avg_acc_rate)
                print(f"Model Path: {reasoner_model_path}, Accuracy: {avg_acc_rate}")
            else:
                avg_acc_rate = evaluate_model(0, 1, reasoner_model_path, critic_model_path)
                results.append(avg_acc_rate)

        # 为当前critic模型绘制结果
        critic_name = critic_model_path.split('/')[-1]
        plt_title = f'Model Performance Comparison (Critic: {critic_name})'
        plot_results(plt_title, [path.split('/')[-1] for path in reasoner_model_paths], results, f'model_performance_{critic_name}.png')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()