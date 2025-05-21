import json
import os
from tqdm import tqdm

def transform(input_path, output_path):
    # 1. 读取原始 JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 构造新的结构
    out = {
        'instance_id': os.path.splitext(os.path.basename(input_path))[0],  # 去掉扩展名
        'problem': data['problem_statement'][0],
        'roles': {},  # 先创建空的roles字典
    }
    
    # history：保留每一次对话的 role 和 content
    hist = []
    for turn in data['trajectory']:
        # content 可能是列表，这里合并为一段字符串
        content = turn['content']
        if isinstance(content, list):
            txt = '\n'.join(content)
        else:
            txt = content
        hist.append({
            'role': turn['role'],
            'name': turn['name'],
            'content': txt
        })
    out['history'] = hist
    
    # Extract roles from history, excluding system messages
    seen = set()
    for turn in hist:
        name = turn.get('name')
        if name and name not in ['Code Execution', 'Error', 'System']:
            seen.add(name)

    # Fill in roles
    for i, role in enumerate(sorted(seen), 1):
        out['roles'][f'agent{i}'] = role
        
    # 设置其他字段
    out['label'] = data['other_data'].get('given', '')
    out['correct'] = data['other_data'].get('correct', False)
    out['annotation'] = {
        "Fail to detect ambiguities/contradictions": None,
        "Proceed with incorrect assumptions": None,
        "Redundant conversation turns for iterative tasks rather than batching": None,
        "No attempt to verify outcome": None,
        "Blurring roles": None
    }

    # 3. 保存到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def process_directory(input_dir, output_dir):
    """处理目录中的所有JSON文件，保持目录结构"""
    for root, _, files in os.walk(input_dir):
        json_files = [f for f in files if f.endswith('.json')]
        if not json_files:
            continue
            
        # 计算相对路径，保持目录结构
        rel_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # 处理当前目录下的所有JSON文件
        for file in tqdm(json_files, desc=f"Processing {rel_path}"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(target_dir, file)
            try:
                transform(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == '__main__':
    input_directory = 'traces/AG2'
    output_directory = 'processed/AG2'
    process_directory(input_directory, output_directory)
    print(f"转换完成，结果保存在 {output_directory} 目录")
