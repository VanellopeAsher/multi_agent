import json
import os
import re
import re
import json
from tqdm import tqdm

def extract_evaluation_success(text):
    """
    从对话文本中提取Evaluation部分的success值
    参数：
        text: 包含完整对话的字符串
    
    返回：
        success值（布尔值）或None（未找到时）
    """
    # 使用正则表达式定位Evaluation块，匹配完整的JSON对象
    eval_pattern = r'Evaluation\s*({[^}]*(?:}[^}]*)*})'
    match = re.search(eval_pattern, text, re.DOTALL)
    
    if not match:
        print("没有找到Evaluation块")
        return None

    try:
        # 处理可能存在的JSON格式问题
        eval_json = match.group(1)
        
        # 清理JSON字符串
        eval_json = eval_json.strip()
        # 确保JSON对象完整性
        if not eval_json.endswith('}'):
            eval_json += '}'
        
        # 移除多余的逗号
        eval_json = re.sub(r',(\s*})', r'\1', eval_json)
        
        # 解析JSON
        data = json.loads(eval_json)
        return data.get('success')
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{str(e)}")
        print(f"尝试解析的JSON内容：{eval_json}")
        return None
    except Exception as e:
        print(f"其他错误：{str(e)}")
        return None

def extract_question(content):
    # 使用正则分割段落（空行或空白符分隔）
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]

    if len(paragraphs) >= 2:
        return paragraphs[0].split('\n')[1]  # 返回第一段话的第一行
    else:
        return "文件中没有第二段话"

def parse_conversation(text):
    history = []
    current_block = {}
    lines = text.split('\n')
    
    # 对话模式匹配规则
    patterns = {
        "agent_response": re.compile(r"Response from (.*?) Agent"),
        "message_to": re.compile(r"Message to (.*?) Agent"),
        "code_execution": re.compile(r"Code Execution Output"),
        "system_message": re.compile(r"(Entering|Exiting) .* Agent message loop"),
        "user_reply": re.compile(r"Reply from .*? to Supervisor"),
        "error": re.compile(r"Execution failed")
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测关键模式
        if any(pattern.search(line) for pattern in patterns.values()):
            if current_block:
                history.append(current_block)
                current_block = {}

        # 分类处理不同消息类型
        if patterns["agent_response"].search(line):
            agent = patterns["agent_response"].search(line).group(1)
            current_block = {
                "role": "assistant",
                "name": f"{agent} Agent",
                "content": []
            }
        elif patterns["message_to"].search(line):
            agent = patterns["message_to"].search(line).group(1)
            current_block = {
                "role": "user", 
                "name": f"{agent} Agent",
                "content": []
            }
        elif patterns["code_execution"].search(line):
            current_block = {
                "role": "system",
                "name": "Code Execution",
                "content": []
            }
        elif patterns["error"].search(line):
            current_block = {
                "role": "system",
                "name": "Error",
                "content": [line]
            }
        elif patterns["system_message"].search(line):
            current_block = {
                "role": "system",
                "name": "System",
                "content": [line]
            }
        else:
            if current_block.get("content") is not None:
                # 清理代码注释和多余空格
                clean_line = re.sub(r'^#+\s*', '', line)
                if clean_line:
                    current_block["content"].append(clean_line)

    # 合并内容并清理
    for item in history:
        content = '\n'.join(item["content"])
        item["content"] = re.sub(r'\n{2,}', '\n', content).strip()
        del item["content"]
        item["content"] = content  # 保持原始格式

    return history

def transform(input_path, output_path):
    # Read the text file
    with open(input_path, 'r', encoding='utf-8') as f:
        text_content = f.read()

    # 构造新的结构，按指定顺序
    out = {
        'instance_id': os.path.splitext(os.path.basename(input_path))[0],  # 去掉.txt后缀
        'problem': extract_question(text_content),
        'roles': {},  # 先创建空的roles字典
        'history': parse_conversation(text_content),
        'label': None,
        'correct': extract_evaluation_success(text_content),
        'annotation': {
            "Fail to detect ambiguities/contradictions": None,
            "Proceed with incorrect assumptions": None,
            "Redundant conversation turns for iterative tasks rather than batching": None,
            "No attempt to verify outcome": None,
            "Blurring roles": None
        }
    }

    # Extract roles from history, excluding system messages
    seen = set()
    for turn in out['history']:
        name = turn.get('name')
        if name and name not in ['Code Execution', 'Error', 'System']:
            seen.add(name)

    # Fill in roles
    for i, role in enumerate(sorted(seen), 1):
        out['roles'][f'agent{i}'] = role

    # Write the output JSON with proper indentation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # 创建输出目录
    output_dir = 'processed/AppWorld'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有txt文件
    input_dir = 'traces/AppWorld'
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # 使用tqdm创建进度条
    for file in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, file)
        # 保持相同的文件名，但改为.json后缀
        output_filename = os.path.splitext(file)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            transform(input_path, output_path)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    print(f"转换完成，结果保存在 {output_dir} 目录")
