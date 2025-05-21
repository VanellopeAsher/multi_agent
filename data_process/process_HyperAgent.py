import json
import sys
import os
from tqdm import tqdm

def convert_to_target_format(data):
    result = {
        "instance_id": data.get("instance_id", None),
        "problem": data.get("problem_statement", None),
        "roles": {},  # 将在后续步骤中填充
        "history": [],
        "label": None,     # ✅ label 为 null
        "correct": None,   # ✅ correct 为 null
        "annotation": {},  # 将在后续步骤中填充
    }

    # --- 构建 history ---
    current_role = None
    current_name = None
    buffer = []

    for line in data.get("trajectory", []):
        if "- INFO -" in line and "Intern Name:" in line:
            if buffer and current_role and current_name:
                result["history"].append({
                    "agent role": current_role,
                    "agent name": current_name,
                    "content": "\n".join(buffer).strip()
                })
                buffer = []
            current_role = "assistant"
            current_name = line.split("Intern Name:")[1].strip()

        elif "- INFO -" in line and "Planner's Response" in line:
            if buffer and current_role and current_name:
                result["history"].append({
                    "agent role": current_role,
                    "agent name": current_name,
                    "content": "\n".join(buffer).strip()
                })
                buffer = []
            current_role = "assistant"
            current_name = "planner"

        elif "- INFO -" in line and "Executor->Planner" in line:
            if buffer and current_role and current_name:
                result["history"].append({
                    "role": current_role,
                    "name": current_name,
                    "content": "\n".join(buffer).strip()
                })
                buffer = []
            current_role = "assistant"
            current_name = "executor"

        elif "- INFO -" in line:
            continue

        else:
            buffer.append(line.strip())

    if buffer and current_role and current_name:
        result["history"].append({
            "role": current_role,
            "name": current_name,
            "content": "\n".join(buffer).strip()
        })

    # --- 提取 history 中的所有 name 并更新 roles ---
    unique_names = {entry.get("agent name", None) for entry in result["history"] if "agent name" in entry}
    result["roles"] = {f"agent{i+1}": name for i, name in enumerate(unique_names)}

    # --- 过滤 annotation 字段 ---
    allowed_keys = {
        "Fail to detect ambiguities/contradictions",
        "Proceed with incorrect assumptions",
        "Redundant conversation turns for iterative tasks rather than batching",
        "No attempt to verify outcome",
        "Blurring roles"
    }
    original_annotation = data.get("note", {}).get("options", {})
    result["annotation"] = {key: original_annotation.get(key, None) for key in allowed_keys}

    return result


def process_all_json_files(input_dir, output_dir):
    """
    处理指定目录中的所有 JSON 文件，并将结果保存到指定的输出目录。
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    """
    # 获取所有 JSON 文件的路径
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for input_path in tqdm(all_files, desc="Processing JSON files"):
        relative_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        output_path = os.path.join(output_dir, relative_path, os.path.basename(input_path))

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # 读取 JSON 文件
            with open(input_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)

            # 转换数据格式
            result = convert_to_target_format(data)

            # 保存结果到新文件
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(result, outfile, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "traces/HyperAgent"  # 输入目录
    output_directory = "processed/HyperAgent"  # 输出目录
    process_all_json_files(input_directory, output_directory)
