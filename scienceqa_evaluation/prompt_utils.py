"""
提示词相关的工具函数
"""

system_prompt = """
A conversation between User and Assistant. 
The user asks a question, and the Assistant solves it. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags.
"""

def make_question_prompt(question, choices, hint):
    """
    构建ScienceQA问题的用户提示
    
    Args:
        question: 问题文本
        choices: 选项列表
        hint: 提示信息
    
    Returns:
        格式化后的用户提示
    """
    question_prompt = f"Question: {question}\n" + \
                  "".join(f"{idx}: {choice}\n" for idx, choice in enumerate(choices)) + \
                  "Requirement: This is a multiple-choice question. Please select one answer choice based on the question and the provided options.\n" + \
                  f"Hint: {hint}\n"
    return question_prompt


def make_critic_prompt(question, choices, hint, first_reasoning):
    """
    构建评判模型的提示词
    
    Args:
        question: 问题文本
        choices: 选项列表
        hint: 提示信息
        first_reasoning: 第一次推理的结果
    
    Returns:
        评判模型的提示词
    """
    critic_prompt = f"{make_question_prompt(question, choices, hint)}\n" + \
                    f"Answer: {first_reasoning}\n" + \
                    f"Task: Please provide a critique of the answer above. What are the weaknesses of the answer?"
    return critic_prompt


def make_reflection_prompt(question, choices, hint, critic_output):
    """
    构建反思提示词
    
    Args:
        critic_output: 评判模型的输出
        question: 原始问题
    
    Returns:
        反思提示词
    """
    reflection_prompt = f"Reflection on former answer:\n" + \
                        f"{critic_output}\n" + \
                        f"Question: {make_question_prompt(question, choices, hint)}"
    return reflection_prompt
