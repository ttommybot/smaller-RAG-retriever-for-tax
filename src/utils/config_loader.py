# -*- coding: utf-8 -*-
"""
配置加载工具模块

提供统一的配置加载功能，支持本地配置覆盖。
所有需要加载配置的模块应使用此模块的 load_config 函数。
"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "configs/configs.yaml") -> Dict:
    """
    从 YAML 文件加载配置，支持本地配置覆盖。

    加载顺序：
    1. 加载主配置文件 (configs.yaml)
    2. 如果存在 configs_local.yaml，则合并其配置（覆盖主配置）

    参数
    ----------
    config_path : str, optional
        主配置文件路径，默认为 "configs/configs.yaml"。

    返回
    -------
    Dict
        合并后的配置字典。
    """
    # 加载主配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 尝试加载本地配置进行覆盖
    config_dir = Path(config_path).parent
    local_config_path = config_dir / "configs_local.yaml"

    if local_config_path.exists():
        with open(local_config_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        _merge_config(config, local_config)

    return config


def _merge_config(base: Dict, override: Dict) -> None:
    """
    递归合并配置字典（原地修改 base）。

    参数
    ----------
    base : Dict
        基础配置字典（会被修改）。
    override : Dict
        覆盖配置字典。
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print("加载的配置:")
    print(f"  项目名称：{config.get('project_name')}")
    print(f"  Generator backend: {config.get('models', {}).get('generator_backend')}")
    print(f"  API key: {config.get('models', {}).get('huggingface_api_key', '')[:10]}...")
