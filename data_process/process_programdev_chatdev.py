import json
import os
from tqdm import tqdm
import re
import json
import re

def extract_history_from_log(log_content):
    """
    从日志内容中提取对话历史。

    Args:
        log_content (str): 日志文件的内容。

    Returns:
        list: 一个字典列表，每个字典包含 "agent_name" 和 "content"。
    """
    pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO\] ([^:]+): (.*?)(?=\n\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\]|$)', re.DOTALL)
    history = []
    for match in pattern.finditer(log_content):
        agent_name = match.group(2).strip()
        content = match.group(3).strip()
        history.append({'name': agent_name, 'content': content})
    return history
def transform(input_path, output_path):
    log_files = [f for f in os.listdir(input_path) if f.endswith('.log')]
    log_content = ""
    for log_file in log_files:
        log_file_path = os.path.join(input_path, log_file)
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content += f.read()
    prompt_files = [f for f in os.listdir(input_path) if f.endswith('.prompt')]
    question = None
    if prompt_files:
        prompt_file_path = os.path.join(input_path, prompt_files[0])
        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            question = pf.read().strip()
    hist = extract_history_from_log(log_content)
    roles_list = []
    roles = {}
    with open(os.path.join(input_path, 'RoleConfig.json'), 'r', encoding='utf-8') as rf:
        roles_list = json.load(rf).keys()
    i = 0
    for role in roles_list:
        i += 1
        agent = 'agent' + str(i)
        roles[agent] = role
    history = []
    for h in hist:
        if h['name'] in list(roles.values()):
            history.append(h) 
    
    instance_id = os.path.splitext(os.path.basename(input_path))[0].split('_')[0]
    
    out = {
        'instance_id': instance_id,
        'problem': question,
        'roles': roles,
        'history': history,
        'label': None,
        'correct': None,
        'annotation': {
            "Fail to detect ambiguities/contradictions": None,
            "Proceed with incorrect assumptions": None,
            "Redundant conversation turns for iterative tasks rather than batching": None,
            "No attempt to verify outcome": None,
            "Blurring roles": None
        }
    }

    # 保存到指定的 output_path 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path+'.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

def process_directory(input_dir, output_dir):
    log_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    for folder in log_folders:
        input_path = os.path.join(input_dir, folder)
        output_path = os.path.join(output_dir, folder)
        try:
            transform(input_path, output_path)
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")

if __name__ == '__main__':
    input_directory = 'traces/programdev/chatdev'
    output_directory = 'processed/programdev/chatdev'
    process_directory(input_directory, output_directory)
    print(f"转换完成，结果保存在 {output_directory} 目录")
