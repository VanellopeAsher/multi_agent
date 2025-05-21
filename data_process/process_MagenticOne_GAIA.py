import json
import os
from tqdm import tqdm
import re

def parse_history(log_text):
    # 1. 用正则把整体分成若干段，每段以 "---------- name ----------" 开头
    #    保留分隔符中的 name
    pattern = re.compile(r'^-{10}\s*(?P<name>.+?)\s*-{10}\s*$', re.MULTILINE)
    parts = pattern.split(log_text)
    # split 结果为 [前置无关, name1, body1, name2, body2, ...]
    
    history = []
    # 忽略 parts[0]，从 i=1 开始，每两个元素一组 (name, body)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i+1].strip()
        # 去掉重复的空行，并保持原有换行
        lines = [line for line in body.splitlines() if line.strip()]
        content = "\n".join(lines)
        
        history.append({
            "role": "user" if name.lower() == "user" else "assistant",
            "name": name,
            "content": content
        })
    return history

def transform(input_path, output_path):
    print(f"Processing file: {input_path}")
    # 1. 读取原始 JSON
    with open(input_path + '/0/prompt.txt', 'r', encoding='utf-8') as f:
        problem = f.read()
    with open(input_path + '/0/expected_answer.txt', 'r', encoding='utf-8') as f:
        label = f.read().rstrip('\n')  # 去除末尾的换行符
    with open(input_path + '/0/console_log.txt', 'r', encoding='utf-8') as f:
        logs = f.read()

    seen = set()
    agent_list = []
    history = parse_history(logs)
    for entry in history:
        name = entry["name"]
        if name not in seen:
            seen.add(name)
            if name != "user":
                agent_list.append(name)

    # 2. 构造 roles 字典
    roles = {f"agent{i+1}": name for i, name in enumerate(agent_list)}
    m = re.search(r'FINAL ANSWER:\s*(.*)', logs)
    if m:
        final_answer = m.group(1).strip()
        print(final_answer) 
    else:
        print("未找到 FINAL ANSWER")
    # 2. 构造新的结构
    out = {
        'instance_id': os.path.basename(input_path),
        'problem': problem,
        'roles': roles,
        'history': history,
        'label': label,  # 已去除末尾换行符
        'correct': final_answer,
        'annotation': {
            "Fail to detect ambiguities/contradictions": None,
            "Proceed with incorrect assumptions": None,
            "Redundant conversation turns for iterative tasks rather than batching": None,
            "No attempt to verify outcome": None,
            "Blurring roles": None
        }
    }

    # 3. 保存到指定的 output_path 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

def process_directory(input_dir, output_dir):
    print(f"Processing directory: {input_dir}")
    folders = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    for folder in tqdm(folders, desc=f"Processing {input_dir}"):
        input_path = os.path.join(input_dir, folder)
        output_path = os.path.join(output_dir, f"{folder}.json")  # 使用文件夹名作为文件名
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        try:
            transform(input_path, output_path)
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")

if __name__ == '__main__':
    input_directory = 'traces/MagenticOne_GAIA/gaia_validation_level_1__MagenticOne'
    output_directory = 'processed/MagenticOne_GAIA/gaia_validation_level_1__MagenticOne'
    process_directory(input_directory, output_directory)
    input_directory = 'traces/MagenticOne_GAIA/gaia_validation_level_2__MagenticOne'
    output_directory = 'processed/MagenticOne_GAIA/gaia_validation_level_2__MagenticOne'
    process_directory(input_directory, output_directory)
    input_directory = 'traces/MagenticOne_GAIA/gaia_validation_level_3__MagenticOne'
    output_directory = 'processed/MagenticOne_GAIA/gaia_validation_level_3__MagenticOne'
    process_directory(input_directory, output_directory)
    print("转换完成，结果保存在 processed 文件夹中")
