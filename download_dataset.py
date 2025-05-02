import copy
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# dataset = load_dataset("AI4Math/MathVista", cache_dir="datasets/math_vista")

dataset = load_dataset("OpenGVLab/MMT-Bench", cache_dir="datasets/mmt_bench")



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

# print the first example on the testmini set
print(dataset["testmini"][0])
# print(dataset["testmini"][0]['pid']) # print the problem id 
# print(dataset["testmini"][0]['question']) # print the question text 
# print(dataset["testmini"][0]['query']) # print the query text
# print(dataset["testmini"][0]['image']) # print the image path
# print(dataset["testmini"][0]['answer']) # print the answer
# dataset["testmini"][0]['decoded_image'] # display the image

# # print the first example on the test set
# print(dataset["test"][0])
