import copy
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# dataset = load_dataset("AI4Math/MathVista", cache_dir="datasets/math_vista")

dataset = load_dataset("derek-thomas/ScienceQA", cache_dir="datasets/science_qa")



# def make_conversation_image(example):
#         return {
#             "prompt": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image"}, # 要好好找找在哪里插入真正的image的！
#                         {"type": "text", "text": example["question"]},
#                     ],
#                 },
#             ],
#         }


# ds = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
# dataset = dataset.remove_columns(["original_question", "original_answer"])
print("~~~~~~~~~~~~~~~`\n")
for item in dataset["train"]:
    print(item["image"])
