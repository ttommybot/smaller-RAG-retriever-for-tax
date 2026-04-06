# 配置文件说明

## 文件结构

```
configs/
├── configs.yaml           # 主配置文件（可提交到 Git）
├── configs.yaml.template  # 配置模板（参考用）
└── configs_local.yaml     # 本地配置（包含敏感信息，不提交到 Git）
```

## 使用说明

### 1. 首次使用

复制模板文件创建主配置：
```bash
# 如果 configs.yaml 不存在
cp configs/configs.yaml.template configs/configs.yaml
```

### 2. 配置敏感信息

创建或编辑 `configs_local.yaml` 文件，添加你的 API key：
```yaml
models:
  huggingface_api_key: "your_api_key_here"
```

**注意**: `configs_local.yaml` 已被添加到 `.gitignore`，不会被提交到 Git。

### 3. 配置合并规则

程序会自动合并两个文件：
- 首先加载 `configs.yaml` 中的基础配置
- 然后用 `configs_local.yaml` 中的配置覆盖相同字段

例如：
- `configs.yaml` 定义 `generator_backend: "huggingface"`
- `configs_local.yaml` 定义 `huggingface_api_key: "hf_..."`
- 最终配置两者都会包含

## 配置项说明

### paths - 路径配置
| 字段 | 说明 | 默认值 |
|------|------|--------|
| `raw_data_dir` | 原始数据目录 | `data/raw` |
| `processed_data_dir` | 处理数据目录 | `data/processed` |
| `vector_db_dir` | 向量数据库目录 | `vectordb` |

### chunking - 分块配置
| 字段 | 说明 | 默认值 |
|------|------|--------|
| `chunk_size` | 分块大小 | `300` |
| `chunk_overlap` | 分块重叠 | `100` |
| `min_chunk` | 最小分块长度 | `100` |

### embedding - 嵌入模型配置
| 字段 | 说明 | 默认值 |
|------|------|--------|
| `model_large_name` | large 模型名称 | `BAAI/bge-large-zh-v1.5` |
| `model_small_name` | small 模型名称 | `sentence-transformers/all-MiniLM-L6-v2` |
| `model_student_name` | student 模型名称 | `sentence-transformers/all-MiniLM-L6-v2` |

### retrieval - 检索配置
| 字段 | 说明 | 默认值 |
|------|------|--------|
| `top_k` | 检索返回数量 | `5` |

### models - 生成器配置
| 字段 | 说明 | 可选值 |
|------|------|--------|
| `generator_backend` | 生成器后端 | `dummy` / `huggingface` / `openai` / `ollama` |
| `generator_model_name` | 模型名称 | 如 `Qwen/Qwen2.5-7B-Instruct` |
| `huggingface_api_key` | HuggingFace API key | （在 configs_local.yaml 中） |

### app - 应用配置
| 字段 | 说明 | 默认值 |
|------|------|--------|
| `show_sources` | 是否显示来源 | `true` |
