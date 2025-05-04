import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "models/ckpt/qwen2_2b_mathvista_ckpt/checkpoint-300",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = model.eval()
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_scienceqa_user_prompt(question, choices, hint):
    user_prompt = f"Question: {question}\n"
    for idx, choice in enumerate(choices):
        user_prompt += f"{idx}: {choice}\n"
    user_prompt += "Requirement: This is a multiple-choice question. Please select the correct answer based on the question and the provided options.\n"
    user_prompt += f"Hint: {hint}\n"
    return user_prompt
    
def filter_vqa_items(example):
    # Define your condition here. For example, remove items without an image
    return example["image"] is not None and example["image"] != ""

dataset = load_dataset("derek-thomas/ScienceQA", cache_dir="datasets/science_qa")
dataset = dataset.filter(filter_vqa_items) # Apply the filter to the dataset

for example in dataset["test"]:
    ground_truth = example["choices"][example["answer"]]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": SYSTEM_PROMPT + "\n" + make_scienceqa_user_prompt(example["question"], example["choices"], example["hint"])},
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("output_text: ", output_text)
    print("ground_truth: ", ground_truth)
    print()
