# RAG for Tax 项目代码结构

本文档整理了 `configs`、`scripts`、`src` 三个目录下各模块的函数作用、输入输出格式。

---

## 目录结构

```
rag_for_tax/
├── configs/
│   ├── configs.yaml          # 项目配置文件
│   ├── configs.yaml.template # 配置模板
│   ├── configs_local.yaml    # 本地配置（含 API key，不提交到 Git）
│   └── README.md             # 配置说明
├── models/
│   └── sentence-transformers--all-MiniLM-L6-v2/  # 本地 Embedding 模型
├── scripts/
│   ├── ingest.py             # 数据入库脚本
│   ├── query_demo.py         # 查询演示脚本
│   └── download_models.py    # 模型下载脚本
├── src/
│   ├── app/
│   │   └── main.py           # 应用主入口（交互式查询界面）
│   ├── chunking/
│   │   ├── chunker.py        # 文档分块模块
│   │   ├── preprocess.py     # 文本预处理模块
│   │   └── test_chunking.py  # 分块测试脚本
│   ├── embedding/
│   │   ├── embedder.py       # 文本嵌入模块
│   │   ├── vectorstore.py    # 向量数据库模块
│   │   └── test_embedding.py # 嵌入测试脚本
│   ├── generation/
│   │   ├── generator.py      # 答案生成模块（支持 HuggingFace）
│   │   └── prompt_builder.py # 提示词构建模块
│   ├── loading/
│   │   ├── loader.py         # 文档加载模块
│   │   └── test_loading.py   # 加载测试脚本
│   ├── pipeline/
│   │   ├── ingest_pipeline.py # 入库流水线
│   │   └── rag_pipeline.py    # RAG 查询流水线
│   ├── retrieval/
│   │   └── retriever.py      # 检索模块 (使用向量库)
│   └── utils/
│       ├── __init__.py       # 工具包初始化
│       └── config_loader.py  # 配置加载工具
├── vectordb/                  # 向量数据库目录
├── data/
│   ├── raw/                   # 原始文档（Word 格式）
│   └── processed/             # 处理后的数据
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 一、CONFIGS 目录

### configs.yaml - 项目配置文件

**作用**: 存储项目全局配置参数

| 配置项 | 说明 | 值示例 |
|--------|------|--------|
| `project_name` | 项目名称 | "rag_for_tax" |
| `paths.raw_data_dir` | 原始数据目录 | "data/raw" |
| `paths.processed_data_dir` | 处理数据目录 | "data/processed" |
| `paths.vector_db_dir` | 向量数据库目录 | "vectordb" |
| `chunking.chunk_size` | 分块大小 | 300 |
| `chunking.chunk_overlap` | 分块重叠 | 100 |
| `chunking.min_chunk` | 最小分块 | 100 |
| `embedding.model_large_name` | large 模型名称 | "BAAI/bge-large-zh-v1.5" |
| `embedding.model_small_name` | small 模型名称 | "sentence-transformers/all-MiniLM-L6-v2" |
| `embedding.model_student_name` | student 模型名称 | "sentence-transformers/all-MiniLM-L6-v2" |
| `retrieval.top_k` | 检索返回数量 | 5 |
| `models.generator_backend` | 生成器后端 | "huggingface" / "dummy" |
| `models.generator_model_name` | 生成器模型名称 | "Qwen/Qwen2.5-7B-Instruct" |
| `models.huggingface_api_key` | HuggingFace API Key | (在 configs_local.yaml 中) |
| `app.show_sources` | 是否显示来源 | true |

### configs_local.yaml - 本地配置（敏感信息）

**作用**: 存储 API Key 等敏感信息，不提交到 Git

```yaml
models:
  huggingface_api_key: "hf_xxx"  # 你的 HuggingFace API Key
```

---

## 二、SCRIPTS 目录

### scripts/ingest.py - 数据入库脚本

**作用**: 执行完整的 RAG 数据处理管道，将文档转换为向量并入库。

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `ingest()` | 执行完整 RAG 数据入库流程 | `data_dir`: str, optional<br>`chunking_strategy`: str, optional<br>`model_type`: str, optional<br>`batch_size`: int, optional<br>`save_path`: str, optional | None |

**CLI 参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 原始数据目录 | config 中的 data/raw |
| `--strategy` | 分块策略 | sliding_window |
| `--model` | embedding 模型类型 | large |
| `--batch-size` | 批量大小 | 32 |
| `--save-path` | 向量库保存路径 | config 中的 vectordb |

**处理流程**:
1. 加载文档 (data/raw)
2. 文档分块 (滑动窗口/语义)
3. 文本预处理 (清洗、标准化)
4. 生成 Embedding 向量
5. 保存向量数据库

---

### scripts/download_models.py - 模型下载脚本

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `download_model()` | 下载模型到本地 models 目录 | `model_name`: str<br>`save_dir`: str | Path: 保存路径 |

---

### scripts/query_demo.py - 查询演示脚本

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `main()` | 交互式查询主循环 | None | None |

**功能**: 提供交互式命令行查询界面，输入问题后返回 RAG 答案和来源。

---

## 三、SRC 目录

### src/utils/config_loader.py - 配置加载工具

**作用**: 统一配置加载，支持本地配置覆盖

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `load_config()` | 加载配置并合并本地配置 | `config_path`: str | Dict |
| `_merge_config()` | 递归合并配置字典 | `base`: Dict, `override`: Dict | None |

---

### src/loading/loader.py - 文档加载模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `load_documents_from_dir()` | 从目录批量加载 Word 文档 | `directory`: str<br>`file_extension`: str | List[Dict] |
| `parse_file_name()` | 解析文件名中的类型和名称 | `file_stem`: str | Tuple[str, str] |

---

### src/chunking/chunker.py - 文档分块模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `sliding_window_chunking()` | 滑动窗口分块 | `documents`: List[Dict]<br>`window_size`: int<br>`step_size`: int<br>`min_chunk`: int | List[Dict] |
| `raw_data_semantic_chunking()` | 基于分隔符的语义分块 | `documents`: List[Dict] | List[Dict] |
| `get_chunking_config()` | 获取分块配置 | None | Dict |
| `save_chunks()` | 保存 chunk 到 JSON 文件 | `chunks`: List[Dict] | str |

---

### src/chunking/preprocess.py - 文本预处理模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `preprocess_chunks()` | 对 chunk 列表进行预处理 | `chunks`: List[Dict] | List[Dict] |
| `get_chunk_stats()` | 获取 chunk 统计信息 | `chunks`: List[Dict] | Dict |

---

### src/embedding/embedder.py - 文本嵌入模块

**模型加载策略**: small/student 模型优先从本地 `models/` 目录加载，不存在则从 HuggingFace 下载。

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `get_embedder()` | 获取指定模型的 embedder 接口 | `model_type`: str | Dict[str, Callable] |
| `load_small_model()` | 加载 small 模型（本地优先） | None | SentenceTransformer |
| `load_student_model()` | 加载 student 模型（本地优先） | None | SentenceTransformer |
| `load_large_model()` | 加载 large 模型（HuggingFace） | None | SentenceTransformer |

---

### src/embedding/vectorstore.py - 向量数据库模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `build_vectorstore()` | 从 chunk 列表构建向量库 | `chunks`: List[Dict]<br>`model_type`: str<br>`batch_size`: int | Dict |
| `load_vectorstore()` | 从磁盘加载向量库 | `model_type`: str | Dict |
| `search()` | 基于查询文本检索相似 chunk | `query`: str<br>`vectorstore`: Dict<br>`top_k`: int | List[Tuple[Dict, float]] |

---

### src/retrieval/retriever.py - 检索模块

**注意**: 此模块为 Mock 实现，实际检索使用 `embedding/vectorstore.py` 的 `search()` 函数。

---

### src/generation/prompt_builder.py - 提示词构建模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `build_rag_prompt()` | 构造 RAG 专属 Prompt | `query`: str<br>`retrieved_context`: str | str |

---

### src/generation/generator.py - 答案生成模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `generate_answer()` | 根据 Prompt 生成回答 | `prompt`: str | str |

**支持的后端**:
- `dummy`: 测试模式，返回固定回答
- `huggingface`: 调用 HuggingFace Inference API（使用 Qwen/Qwen2.5-7B-Instruct）

---

### src/pipeline/ingest_pipeline.py - 入库流水线模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `run_ingestion_pipeline()` | 执行完整入库流水线 | `config_path`: str | int |

---

### src/pipeline/rag_pipeline.py - RAG 查询流水线模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `run_rag_pipeline()` | 运行完整 RAG 查询流水线 | `query`: str<br>`config_path`: str, optional<br>`model_type`: str | Dict |

**返回格式**:
```python
{
    "answer": "模型回答",
    "sources": ["来源 1", "来源 2", ...]
}
```

---

### src/app/main.py - 应用主入口

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `main()` | 应用主函数（交互式查询循环） | None | None |

---

## 四、数据流向图

```
入库流程:
原始文档 (data/raw/*.docx)
        ↓
[loading/loader.py] load_documents_from_dir()
        ↓
文档列表 [{full_text, file_name, file_type, ...}]
        ↓
[chunking/chunker.py] sliding_window_chunking()
        ↓
Chunk 列表 [{id, content, metadata}]
        ↓
[chunking/preprocess.py] preprocess_chunks()
        ↓
清洗后 Chunk 列表
        ↓
[embedding/embedder.py] get_embedder('small')['embed_texts']()
        ↓
向量矩阵 (n_chunks, embedding_dim)
        ↓
[embedding/vectorstore.py] build_vectorstore()
        ↓
向量数据库 (vectordb/)


查询流程:
用户查询
        ↓
[embedding/embedder.py] embed_query() → 查询向量
        ↓
[embedding/vectorstore.py] search() → Top-K Chunks + 相似度分数
        ↓
格式化上下文（带来源标注）
        ↓
[generation/prompt_builder.py] build_rag_prompt()
        ↓
[generation/generator.py] generate_answer() → HuggingFace API
        ↓
答案 + 来源
```

---

## 五、快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型（可选，会加速首次运行）
```bash
python scripts/download_models.py
```

### 3. 配置 API Key
编辑 `configs/configs_local.yaml`:
```yaml
models:
  huggingface_api_key: "hf_xxx"
```

### 4. 向量化知识库
```bash
python scripts/ingest.py --model small
```

### 5. 运行查询
```bash
python src/app/main.py
```
