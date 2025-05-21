from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json
import re
import os
import random
from datetime import datetime
from datasets import load_dataset
import logging
from sympy import symbols, Eq, solve
NUM_SAMPLE = 10

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename=f'logs/mathchat_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "sk-sZxScCcTfpoDWTEBD6A4D8B00a544c2bB69bF14f8c358976",  
            "base_url": "https://api3.apifans.com/v1"
        }
    ]
}
code_execution_config = {"use_docker": False}

expert = AssistantAgent(
    name="Math_Expert",
    llm_config=llm_config,
    system_message="""
Let’s use Python to solve a math problem.
    Query requirements:
    You should always use the 'print' function for the output and use fractions/radical forms
    instead of decimals.
    You can use packages like sympy to help you.
    You must follow the formats below to write your code:
    '''python
    # your code
    '''
    First state the key idea to solve the problem. You may choose from three ways to solve the
    problem:
    Case 1: If the problem can be solved with Python code directly, please write a program to
    solve it. You can enumerate all possible arrangements if needed.
    Case 2: If the problem is mostly reasoning, you can solve it by yourself directly.
    Case 3: If the problem cannot be handled in the above two ways, please follow this process
    :
    1. Solve the problem step by step (do not over-divide the steps).
    2. Take out any queries that can be asked through Python (for example, any calculations or
    equations that can be calculated).
    3. Wait for me to give the results.
    4. Continue if you think the result is correct. If the result is invalid or unexpected,
    please correct your query or reasoning.
    After all the queries are run and you get the answer, put the answer in \\boxed{}.
    
""",
    code_execution_config=code_execution_config
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=code_execution_config
)

group_chat = GroupChat(
    agents=[user_proxy, expert],
    messages=[],
    max_round=10,
    speaker_selection_method='round_robin', 
    allow_repeat_speaker=False
)
chat_manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

def extract_answer(content: str):
    if not content:
        return None

    boxed = re.search(r'\\boxed\{([^\}]+)\}', content)
    if boxed:
        return boxed.group(1)

    frac = re.search(r'\d+\s*/\s*\d+', content)
    if frac:
        return eval(frac.group()) 

    list_match = re.search(r'\[([^\]]+)\]', content)
    if list_match:
        try:
            return eval(list_match.group(1).strip())[0] 
        except Exception:
            pass

    number = re.search(r'\d+(?:\.\d+)?', content)
    if number:
        return number.group()

    return None

def run_mathchat(problem: str) -> str:
    logger.info(f"开始新问题：{problem}")
    user_proxy.reset()
    chat_manager.groupchat.messages.clear()

    def termination_check(msg_list):
        if msg_list:
            last_msg = msg_list[-1]
            if last_msg["name"] == "User" and not last_msg["content"].strip():
                return True
            if last_msg["name"] == "Math_Expert" and "\\boxed{" in last_msg["content"]:
                return True
        return False

    try:
        user_proxy.initiate_chat(
            chat_manager,
            message=f"Let's solve this math problem together. Problem: {problem}",
            callback=termination_check
        )

        for idx, msg in enumerate(chat_manager.groupchat.messages, 1):
            logger.info(f"Round {idx}")
            logger.info(f"{msg['name']}: {msg['content']}")
            logger.info("-" * 50)

        for msg in reversed(chat_manager.groupchat.messages):
            if msg["name"] == "Math_Expert":
                result = extract_answer(msg["content"])
                if result:
                    logger.info(f"提取答案: {result}")
                    return result

        logger.warning("未能提取答案")
        return "N/A"

    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        return "N/A"

def run_mathchat_with_retry(problem: str, max_retries: int = 3) -> str:
    """运行 mathchat 并在失败时重试"""
    for attempt in range(max_retries):
        try:
            result = run_mathchat(problem)
            if result != "N/A":
                # 尝试转换结果，确保是有效的数值
                float(eval(result))
                return result
        except Exception as e:
            logger.warning(f"第 {attempt + 1} 次尝试失败: {str(e)}")
            continue
    
    logger.error(f"在 {max_retries} 次尝试后仍未获得有效结果")
    return "N/A"

# 加载数据集
dataset = load_dataset("qintongli/GSM-Plus")
valid_samples = [sample for sample in dataset["test"] if sample.get("answer") not in [None, "None", ""]]
samples = random.sample(valid_samples, min(NUM_SAMPLE, len(valid_samples)))

# 批量评测
correct = 0
results = []
os.makedirs("results", exist_ok=True)
processed_count = 0

while processed_count < NUM_SAMPLE:
    for item in samples:
        if processed_count >= NUM_SAMPLE:
            break
            
        gold = item["answer"]
        try:
            gold_float = float(gold)
        except (ValueError, TypeError):
            logger.warning(f"跳过无效的标准答案: {gold}")
            continue

        pred = run_mathchat_with_retry(item["question"])
        try:
            pred_float = float(eval(pred))
        except Exception:
            logger.warning(f"跳过无效的预测结果: {pred}")
            continue

        is_correct = abs(gold_float - pred_float) < 1e-3
        if is_correct:
            correct += 1
        
        result = {
            "question": item["question"],
            "expected": gold,
            "predicted": pred,
            "correct": is_correct
        }
        results.append(result)
        processed_count += 1
        
        # 每次评测后保存结果
        with open("results/mathchat_baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Q: {item['question']}\nPredicted: {pred}, Expected: {gold}")
        print(f"Current Progress: {processed_count}/{NUM_SAMPLE}, Current Accuracy: {(correct/processed_count)*100:.2f}%\n")
        
        # 如果当前样本集处理完但还没有达到目标数量，重新采样
        if processed_count < NUM_SAMPLE and all(samples.index(item) >= len(samples)-1 for item in samples):
            remaining = NUM_SAMPLE - processed_count
            new_samples = random.sample(valid_samples, min(remaining * 2, len(valid_samples)))
            samples.extend(new_samples)

# 最终准确率
print(f"\nFinal Accuracy: {(correct/NUM_SAMPLE) * 100:.2f}% ({correct}/{NUM_SAMPLE})")
print("Results have been saved to results/mathchat_baseline_results.json")