# MathChat 多代理系统复现与自动评测
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json
import re
import random
from datasets import load_dataset
import logging
import os
from datetime import datetime

NUM_SAMPLE = 10

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename=f'logs/mathchat_NewTopology_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": 
    [{"model": "gpt-4", 
    "api_key": 'sk-sZxScCcTfpoDWTEBD6A4D8B00a544c2bB69bF14f8c358976',
    "base_url": "https://api3.apifans.com/v1"}],
}
code_execution_config = {
    "use_docker": False
}

problem_solver = AssistantAgent(
    name="ProblemSolver",
    llm_config=llm_config,
    system_message="""
    You are Agent Problem Solver, and your role is to collaborate with other agents to address
    various challenges.
    For each problem, please follow these steps:
    1. **Document Your Solution**: Write your solution step by step, ensuring it is
    independent of the solutions provided by other agents.
    2. **Engage in Discussion**: Once you have outlined your solution, discuss your approach
    and findings with the other agents.""",
    code_execution_config=code_execution_config
)

coder = AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""
    You are Agent Code Executor. You can solve problems only writing commented Python code.
    For each problem, please follow these steps:
    1. **Develop Your Solution**: Write your solution in Python code, detailing each step
    independently from the solutions provided by other agents.
    2. **Utilize SymPy**: Feel free to use the SymPy package to facilitate calculations and
    enhance your code’s efficiency.
    3. **Display Results**: Ensure that you **print the final result at the end of your Python
    code** (e.g., ‘print(_result_)‘).
    4. **Engage in Discussion**: After obtaining the result from your Python code, discuss
    your findings with the other agents.
    Always format your Python code within:
    '''python
    # your code here
    print(_result_)
    '''
    If you wish to execute your code, please indicate this by stating "SUGGESTED NEXT SPEAKER:
    Agent Code Executor" at the end of your message.

    """
    ,
    code_execution_config=code_execution_config
)

verifier = AssistantAgent(
    name="Verifier",
    llm_config=llm_config,
    system_message="""
    You are Agent Verifier.
    Your role is to critically evaluate the solutions proposed by other agents step by step
    and provide a final solution.
    1. **Solution Requirement**: Before making any decisions, ensure you have received
    solutions from both Agent Code Executor and Agent Problem Solver. If either proposed
    solution is missing, do not draw any conclusions; instead, suggest the next speaker by
    stating: SUGGESTED NEXT SPEAKER: _suggested_agent_name_.
    2. **Avoid Assumptions**: Pay attention to the variables provided in the original problem
    statement versus those assumed by the agents. **Assumed values are not valid for the
    solution** and can lead to inaccuracies. Never base your solution on assumed values.
    Always base your solution on the explicitly given variables to ensure correctness. If
    a problem is deemed unsolvable due to missing information, return: **SOLUTION_FOUND \\
    boxed{'None'}**.
    3. **Evaluating Conflicting Solutions**: If different answers are presented during the
    discussion, choose the most appropriate solution based on your evidence or initiate
    further discussion to clarify.
    4. **Final Solution Declaration**: When you are confident about the final solution, return
    it as follows: **SOLUTION_FOUND \\boxed{_solution_value_here_}**. Ensure that only
    numerical values are placed inside the \\boxed{}; any accompanying text should be
    outside.

    """
    ,
    code_execution_config=code_execution_config
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=code_execution_config
)

group_chat = GroupChat(
    agents=[user_proxy, problem_solver, coder, verifier],
    messages=[],
    max_round=10
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
        return eval(frac.group())  # 将 '3/2' 转为 1.5

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

    try:
        user_proxy.initiate_chat(
            chat_manager,
            message=f"Let's solve this math problem together. Problem: {problem}"
        )
        for idx, msg in enumerate(chat_manager.groupchat.messages, 1):
            logger.info(f"Round {idx}")
            logger.info(f"{msg['name']}: {msg['content']}")
            logger.info("-" * 50)
        for msg in reversed(chat_manager.groupchat.messages):
            if msg["name"] == "Verifier":
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
        with open("results/mathchat_NewTopology_results.json", "w") as f:
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
print("Results have been saved to results/mathchat_NewTopology_results.json")