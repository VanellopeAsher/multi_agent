"""
This file contains the constants used in the package.
"""

from typing import Literal


MAP_DATA_PATH = {
    'Chicago': 'cache/Chicago/map_data_Chicago.pkl',
    'Dallas': 'cache/Dallas/map_data_Dallas.pkl',
    "LA": 'cache/LA/map_data_LA.pkl',
    'NY': 'cache/NY/map_data_NewYork.pkl'
}
# Define the map scope of the each city
MAP_SCOPE = {
    'Chicago': [
        {
            'left_bottom': [-87.68478590505254, 41.838495667271154],
            'right_top': [-87.60719496495767, 41.91339680181598]
        },
        {
            'left_bottom': [-87.6453500684818, 41.877741000000004],
            'right_top': [-87.61445102154136, 41.90170819765155],
        },
        {
            'left_bottom': [-87.74390330467695, 41.84420634752652],
            'right_top': [-87.64917529757193, 41.91094190880981],
        },
    ],
    'Dallas': [
        {
            'left_bottom': [-97.000482, 32.613216],
            'right_top': [-96.4636317, 33.0239366]
        }
    ],
    'LA': [
        {
            'left_bottom': [-118.6681798, 33.659541],
            'right_top': [-118.1552983, 34.337306]
        }
    ],
    'NY': [
        {
            'left_bottom': [-74.258843, 40.476578],
            'right_top': [-73.700233, 40.91763]
        }
    ]
}

# Define the type of the platform
PLATFORM = Literal['openai', 'siliconflow', 'deepseek', 'zhipu', 'aliyun']

# Define the API key environment variables 
API_KEY_MAP = {
    'openai': 'OPENAI_API_KEY',
    'siliconflow': 'SILICONFLOW_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'zhipu': 'ZHIPU_API_KEY',
    'aliyun': 'ALIYUN_API_KEY'
}

# Define the base URL for each platform
BASE_URL_MAP = {
    'openai': None, 
    'siliconflow': 'https://api.siliconflow.cn/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4/',
    'aliyun': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
}

MODEL_LIST = {
    'openai': [
        'gpt-4o-mini',  # OpenAI GPT-4o Mini
        'gpt-4o',       # OpenAI GPT-4o 
    ],
    'siliconflow': [
        "Qwen/Qwen2.5-7B-Instruct",     # free, qwen-2.5-7b-instruct
        "Pro/Qwen/Qwen2.5-7B-Instruct", # ￥0.35/ M token, qwen-2.5-7b-instruct
        "Qwen/Qwen2.5-14B-Instruct",    # ￥0.75/ M token, qwen-2.5-14b-instruct
        "Qwen/Qwen2.5-32B-Instruct",    # ￥1.26/ M token, qwen-2.5-32b-instruct
        "Qwen/Qwen2.5-72B-Instruct",    # ￥4.13/ M token, qwen-2.5-64b-instruct
        "meta-llama/Meta-Llama-3.1-8B-Instruct",  # free, meta-llama-3.1-8b-instruct
        "Pro/meta-llama/Meta-Llama-3.1-8B-Instruct", # ￥0.42/ M token, meta-llama-3.1-8b-instruct
        "meta-llama/Meta-Llama-3.1-70B-Instruct", # ￥4.13/ M token, meta-llama-3.1-70b-instruct
        "meta-llama/Meta-Llama-3.1-405B-Instruct" # ￥21.0/ M token, meta-llama-3.1-405b-instruct
        "BAAI/bge-m3",  # free, bge-m3, embedding model
        "deepseek-ai/DeepSeek-V2.5", # ￥1.33/ M token, deepseek-v2.5
    ],
    'deepseek': [
        "deepseek-chat",    # ￥1.0 / M token, deepseek-chat, cache 0.1
        "deepseek-v2.5",    # ￥1.0 / M token, deepseek-v2.5
    ],
    'zhipu': [],
    'aliyun': []
}