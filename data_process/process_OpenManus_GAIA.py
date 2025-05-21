import re
import json
import os
import re
def extract_history_from_log(log_content):
    """
    使用替代方法从INFO日志行中提取 "agent_name" 和 "content"。
    此方法通过定位所有日志条目的起始点来确定每个条目的范围。

    Args:
        log_content (str): 日志文件的原始字符串内容。

    Returns:
        list: 一个字典列表，每个字典包含 "agent_name" 和 "content"。
    """
    extracted_items = []
    
    # 1. 找到所有 "时间戳 | INFO     | " 前缀的起始和结束位置
    #    match.start(0) 是 "时间戳..." 的开始
    #    match.end(0) 是 "... | INFO     | " 这个前缀结束之后的位置
    entry_prefix_matches = list(re.finditer(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| INFO\s*\| ",  # 注意末尾的空格
        log_content,
        re.MULTILINE
    ))
    
    num_entries = len(entry_prefix_matches)
    if num_entries == 0:
        return []

    for i in range(num_entries):
        current_match = entry_prefix_matches[i]
        
        # full_info_part 的开始是当前匹配前缀的结束位置
        start_of_full_info_part = current_match.end(0)
        
        # full_info_part 的结束是下一个日志条目前缀的开始位置，或者是整个日志的末尾
        if i + 1 < num_entries:
            end_of_full_info_part = entry_prefix_matches[i+1].start(0)
        else:
            end_of_full_info_part = len(log_content)
            
        # 提取 full_info_part (这包含了 agent 标识和实际的多行消息)
        # 这个切片操作会获取从当前 INFO 条目的消息体开始，到下一个时间戳条目（或文件尾）之前的所有内容
        full_info_part_text = log_content[start_of_full_info_part:end_of_full_info_part].strip()
        
        # 应用规则：按第一个冒号分割
        parts = full_info_part_text.split(':', 1)
        
        if len(parts) == 2:
            agent_name = parts[0].strip()
            content = parts[1].strip() # content 部分保留其内部的多行结构和空行
            
            if agent_name:
                extracted_items.append({
                    "role": 'assistant',
                    "name": agent_name,
                    "content": content
                })
        # else: full_info_part_text 中没有冒号，不符合分割规则

    return extracted_items

def transform(input_path, output_path):
    # 初始化 out 以避免未定义错误
    out = {
        'instance_id': None,
        'problem': None,
        'roles': {},
        'history': [],
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

    hist = extract_history_from_log(log_content=open(input_path, 'r', encoding='utf-8').read())
    roles = {}
    seen = set()
    for turn in hist:
        name = turn.get('name')
        if name and name not in ['Code Execution', 'Error', 'System']:
            seen.add(name)

    # Fill in roles
    for i, role in enumerate(sorted(seen), 1):
        roles[f'agent{i}'] = role

    # 同时查找 level_1、level_2、level_3 的参考 json 文件，优先级依次为 1 > 2 > 3
    instance_id = os.path.splitext(os.path.basename(input_path))[0]
    problem = None
    label = None
    for level in [1, 2, 3]:
        ref_json_path = os.path.join(
            'processed', 'MagenticOne_GAIA', f'gaia_validation_level_{level}__MagenticOne', f'{instance_id}.json'
        )
        if os.path.exists(ref_json_path):
            with open(ref_json_path, 'r', encoding='utf-8') as ref_f:
                ref_data = json.load(ref_f)
                problem = ref_data.get('problem')
                label = ref_data.get('label')
            break  # 找到优先级最高的就停止

    # 用读取到的 problem 和 label 覆盖 out 里的默认值
    out['problem'] = problem
    out['label'] = label
    out['instance_id'] = instance_id
    out['roles'] = roles
    out['history'] = hist

    # 保存到指定的 output_path 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

def process_directory(input_dir, output_dir):
    """处理目录中的所有日志文件，保持目录结构"""
    log_files = [f for f in os.listdir(input_dir)]
    for file in log_files:
        input_path = os.path.join(input_dir, file)
        output_filename = os.path.splitext(file)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        try:
            transform(input_path, output_path)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == '__main__':
    input_directory = 'traces/OpenManus_GAIA'
    output_directory = 'processed/OpenManus_GAIA'
    process_directory(input_directory, output_directory)
    print(f"转换完成，结果保存在 {output_directory} 目录")
