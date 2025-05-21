import os
import json
from typing import List, Dict, Any
import pandas as pd
import json
import time

def extract_trajectory_to_string(json_data_str: str) -> str:
    """
    从 JSON 字符串中提取轨迹 (trajectory) 并将其格式化为单个字符串。

    Args:
        json_data_str: 包含 JSON 数据的字符串。

    Returns:
        一个表示格式化后轨迹的字符串。
    """
    try:
        data = json.loads(json_data_str)
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {e}"

    if "trajectory" not in data or not isinstance(data["trajectory"], list):
        return "错误: 未找到 'trajectory' 键或其不是一个列表。"

    trajectory_parts = []
    for entry in data["trajectory"]:
        role = entry.get("role", "Unknown role")
        name = entry.get("name", "Unknown name")
        content_list = entry.get("content", [])

        # 将内容列表中的每一行连接起来，保留空字符串（通常代表换行）
        # 同时处理 content_list 中可能存在的 None 或非字符串项
        formatted_content_lines = []
        for line in content_list:
            if line is None:
                formatted_content_lines.append("") # 将 None 视为空行
            elif isinstance(line, str):
                formatted_content_lines.append(line)
            else:
                formatted_content_lines.append(str(line)) # 将非字符串转换为字符串

        content_str = "\n".join(formatted_content_lines)
        
        # 为每个发言者/角色的条目添加一个头部
        speaker_header = f"[{role.strip()} - {name.strip()}]:"
        trajectory_parts.append(f"{speaker_header}\n{content_str}")

    # 使用两个换行符连接所有轨迹条目，以便更好地区分
    return "\n\n".join(trajectory_parts)

def parse_responses(trajectory: str) -> List[str]:
    """
    将轨迹字符串解析为个别角色的回答列表
    
    Args:
        trajectory: 包含完整对话历史的字符串
        
    Returns:
        List[str]: 包含每个主要角色完整回答的列表
    """
    responses = []
    current_response = []
    current_role = None
    
    for line in trajectory.split('\n'):
        # 检查是否是新角色的开始
        if line.startswith('[') and '] - ' in line and line.endswith(':'):
            if current_role and current_response:
                responses.append('\n'.join(current_response))
                current_response = []
            current_role = line
        else:
            current_response.append(line)
            
    # 添加最后一个响应
    if current_response:
        responses.append('\n'.join(current_response))
        
    return responses

def read_ag2_data(folder_path: str) -> List[Dict[str, Any]]:
    """读取AG2文件夹中的数据"""
    data_list = []
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 提取问题和对话历史
                    query = data.get('problem_statement', [''])[0]
                    trajectory_str = extract_trajectory_to_string(json.dumps(data))
                    responses = parse_responses(trajectory_str)
                    
                    # 添加到数据列表
                    data_list.append({
                        'query': query,
                        'responses': responses,
                        'source_file': file,
                    })
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
    
    return data_list

def save_marm_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save MARM evaluation results to a JSON file
    """
    try:
        output_dict = {
            "metadata": {
                "total_entries": len(results),
                "timestamp": str(time.strftime("%Y-%m-%d %H:%M:%S"))
            },
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
