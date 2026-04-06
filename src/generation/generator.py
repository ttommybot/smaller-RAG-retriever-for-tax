# -*- coding: utf-8 -*-
"""
答案生成模块

根据组装好的 Prompt 调用大模型生成回答。
支持的后端：
- dummy: 测试模式，返回固定回答
- aliyun: 调用阿里云 DashScope API（通义千问）
"""

from pathlib import Path
import sys
from typing import Optional

# 添加当前目录到路径以便导入
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from utils.config_loader import load_config


def generate_answer(prompt: str, config_path: str = "configs/configs.yaml") -> str:
    """
    根据组装好的 Prompt 调用大模型生成回答。

    支持的后端：
    - dummy: 测试模式，返回固定回答
    - aliyun: 调用阿里云 DashScope API（通义千问）

    参数
    ----------
    prompt : str
        组装好的 RAG Prompt。
    config_path : str, optional
        配置文件路径，默认为 "configs/configs.yaml"。

    返回
    -------
    str
        模型生成的回答。
    """
    config = load_config(config_path)
    backend = config.get("models", {}).get("generator_backend", "dummy")

    if backend == "dummy":
        return "[Dummy 模式返回]: 这是一个测试回答。我收到了你的 Prompt，但我现在是个假模型，所以只能给你返回这句废话。"

    elif backend == "aliyun":
        return _generate_aliyun_answer(
            prompt,
            config.get("models", {}).get("generator_model_name", "qwen-max"),
            config.get("models", {}).get("aliyun_dashscope_api_key", "")
        )

    else:
        return f"不支持的 backend: {backend}"


def _generate_aliyun_answer(prompt: str, model_name: str, api_key: str) -> str:
    """
    调用阿里云 DashScope API 生成回答（通义千问）。

    参数
    ----------
    prompt : str
        组装好的 RAG Prompt。
    model_name : str
        模型名称，如 "qwen-max"、"qwen-plus"、"qwen-turbo"。
    api_key : str
        阿里云 DashScope API Key。

    返回
    -------
    str
        模型生成的回答。
    """
    import requests

    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        # 阿里云 DashScope API 返回格式
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
        else:
            return f"API 返回格式异常：{result}"

        return generated_text.strip()

    except requests.exceptions.Timeout:
        return "请求超时，请稍后重试。"
    except requests.exceptions.RequestException as e:
        return f"API 请求失败：{str(e)}"
    except Exception as e:
        return f"生成回答时出错：{str(e)}"


# 测试
if __name__ == "__main__":
    from generation.prompt_builder import build_rag_prompt

    test_prompt = build_rag_prompt(
        "什么是增值税？",
        "增值税是以商品在流转过程中产生的增值额作为计税依据而征收的一种流转税。"
    )
    answer = generate_answer(test_prompt)

    print("\n--- 大模型回答 ---")
    print(answer)
