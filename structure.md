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
│   ├── configs_local.yaml.template # 本地配置模板
│   └── README.md             # 配置说明
├── models/
│   ├── sentence-transformers--all-MiniLM-L6-v2/  # MiniLM 小型嵌入模型
│   ├── BAAI--bge-large-zh-v1.5/                  # BGE 大型中文嵌入模型
│   ├── BAAI--bge-reranker-v2-gemma/              # BGE 重排序模型
│   └── README.md                                 # 模型目录说明
├── scripts/
│   ├── chunk_to_processed.py # 文档分块脚本（loading -> chunking -> preprocessing）
│   ├── embed_to_vectordb.py  # 向量化入库脚本（embedding -> vectorstore）
│   ├── download_models.py    # 模型下载脚本
│   └── query_demo.py         # 查询演示脚本
├── src/
│   ├── app/
│   │   └── main.py           # 应用主入口（命令行交互式查询界面）
│   ├── chunking/
│   │   ├── chunker.py        # 文档分块模块
│   │   ├── preprocess.py     # 文本预处理模块
│   │   └── test_chunking.py  # 分块测试脚本
│   ├── embedding/
│   │   ├── embedder.py       # 文本嵌入模块
│   │   ├── vectorstore.py    # 向量数据库模块
│   │   └── test_embedding.py # 嵌入测试脚本（支持 full/bge 模式）
│   ├── generation/
│   │   ├── generator.py      # 答案生成模块（支持阿里云/钉钉）
│   │   └── prompt_builder.py # 提示词构建模块
│   ├── "Interactive interface"/
│   │   ├── app.py            # Web 应用后端（Flask API）
│   │   ├── index.html        # 前端界面
│   │   ├── script.js         # 前端交互逻辑
│   │   ├── run.bat           # Windows 启动脚本
│   │   └── README.md         # 模块说明文档
│   ├── loading/
│   │   ├── loader.py         # 文档加载模块
│   │   └── test_loading.py   # 加载测试脚本
│   ├── pipeline/
│   │   ├── ingest_pipeline.py # 入库流水线
│   │   └── rag_pipeline.py    # RAG 查询流水线
│   ├── query/
│   │   ├── clean.py           # 问题清洗模块
│   │   └── generate_dataset.py # 问题数据集生成模块
│   ├── retrieval/
│   │   ├── retriever.py      # 检索模块（基于向量库）
│   │   └── test_retriever.py # 检索测试脚本
│   └── utils/
│       ├── __init__.py       # 工具包初始化
│       └── config_loader.py  # 配置加载工具
├── vectordb/                  # 向量数据库目录
├── data/
│   ├── raw/                   # 原始文档（Word 格式）
│   ├── processed/             # 处理后的数据
│   └── query/                 # 生成的问题数据集
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
| `models.generator_backend` | 生成器后端 | "aliyun" / "dummy" |
| `models.generator_model_name` | 生成器模型名称 | "qwen-max" |
| `app.show_sources` | 是否显示来源 | true |
| `query.input_file` | 原始问题文件 | "seed_questions.txt" |
| `query.output_file` | 干净问题文件 | "data/query/seed_questions_cleaned.txt" |
| `query.seed_file` | 种子问题文件 | "data/query/seed_questions_cleaned.txt" |
| `query.output_dir` | 问题集输出目录 | "data/query" |
| `query.generate_per_seed` | 每条种子生成数量 | 6 |
| `query.concurrent_limit` | API 并发限制 | 15 |
| `query.dataset_sizes` | 数据集规模 | {small: 200, medium: 500, large: 1000} |

### configs_local.yaml - 本地配置（敏感信息）

**作用**: 存储 API Key 等敏感信息，不提交到 Git

```yaml
# 生成器 API 配置
models:
  generator_backend: "aliyun"
  generator_model_name: "qwen-max"
  aliyun_dashscope_api_key: "sk-xxx"

# 问题集生成 API 配置
api:
  generation_url: "https://api.closeai-asia.com/v1/chat/completions"
  generation_api_key: "sk-xxx"
  generation_model: "deepseek-chat"
```

---

## 二、SCRIPTS 目录

### scripts/chunk_to_processed.py - 文档分块脚本

**执行分块流程**：loading -> chunking -> preprocessing，输出两种分块方法的 cleaned JSON 文件到 data/processed。

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `chunk_to_processed()` | 执行完整分块流程 | `data_dir`: str, optional | None |

**输出文件**:
- `data/processed/chunks_semantic.json` - 语义分块原始结果
- `data/processed/chunks_semantic_cleaned.json` - 语义分块清洗后结果
- `data/processed/chunks_sliding.json` - 滑动窗口分块原始结果
- `data/processed/chunks_sliding_cleaned.json` - 滑动窗口分块清洗后结果

**使用示例**:
```bash
# 使用默认配置
python scripts/chunk_to_processed.py
```

---

### scripts/embed_to_vectordb.py - 向量化入库脚本

**执行向量化流程**：从 data/processed 读取 cleaned JSON，进行 embedding 并存入向量库。

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `embed_to_vectordb()` | 执行向量化入库流程 | `chunk_method`: str<br>`model_type`: str<br>`batch_size`: int<br>`save_path`: str, optional | None |

**支持 6 种组合**：2 种 chunk_method × 3 种 model_type
- `chunk_method`: 'semantic' 或 'sliding_window'
- `model_type`: 'large'、'small'、'student'

**使用示例**:
```bash
# 使用 large 模型 + semantic 分块
python scripts/embed_to_vectordb.py --model large --chunk-method semantic

# 使用 small 模型 + sliding_window 分块
python scripts/embed_to_vectordb.py --model small --chunk-method sliding_window
```

---

### scripts/download_models.py - 模型下载脚本

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `download_model()` | 下载模型到本地 models 目录 | `model_name`: str<br>`save_dir`: str | Path |

---

### scripts/query_demo.py - 查询演示脚本

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `main()` | 交互式查询主循环 | None | None |

---

## 三、SRC 目录

### src/utils/config_loader.py - 配置加载工具

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `load_config()` | 加载配置并合并本地配置 | `config_path`: str | Dict |
| `_merge_config()` | 递归合并配置字典 | `base`: Dict, `override`: Dict | None |

---

### src/query/clean.py - 问题清洗模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `clean_questions()` | 清理问题文件中的日期等内容 | `input_file`: str, optional<br>`output_file`: str, optional<br>`config_path`: str | Tuple[int, str] |

**返回**: (清理的问题数量，输出文件路径)

**配置依赖**:
- `query.input_file`: 输入文件路径
- `query.output_file`: 输出文件路径

---

### src/query/generate_dataset.py - 问题数据集生成模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `generate_dataset()` | 同步接口：生成问题数据集 | `seed_file`: str, optional<br>`output_dir`: str, optional<br>`config_path`: str | Dict |
| `generate_dataset_async()` | 异步接口：批量生成问题数据集 | 同上 | Dict |

**配置依赖**:
- `api.generation_url`: API 端点
- `api.generation_api_key`: API Key
- `api.generation_model`: 生成模型
- `query.seed_file`: 种子文件
- `query.output_dir`: 输出目录
- `query.generate_per_seed`: 每条种子生成数量
- `query.concurrent_limit`: 并发限制
- `query.dataset_sizes`: 数据集规模定义

**输出文件**:
- `data/query/tax_queries_small.txt` (200 条)
- `data/query/tax_queries_medium.txt` (500 条)
- `data/query/tax_queries_large.txt` (1000 条)
- `data/query/README.md`

---

### src/loading/loader.py - 文档加载模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `load_documents_from_dir()` | 从目录批量加载 Word 文档 | `directory`: str<br>`file_extension`: str | List[Dict] |

---

### src/chunking/chunker.py - 文档分块模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `sliding_window_chunking()` | 滑动窗口分块 | `documents`: List[Dict]<br>`window_size`: int<br>`step_size`: int<br>`min_chunk`: int | List[Dict] |
| `raw_data_semantic_chunking()` | 基于分隔符的语义分块 | `documents`: List[Dict] | List[Dict] |

---

### src/chunking/preprocess.py - 文本预处理模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `preprocess_chunks()` | 对 chunk 列表进行预处理 | `chunks`: List[Dict] | List[Dict] |

---

### src/embedding/embedder.py - 文本嵌入模块

**模型加载策略**: small/student 模型优先从本地 `models/` 目录加载，large 模型从 HuggingFace 加载。

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `get_embedder()` | 获取指定模型的 embedder 接口 | `model_type`: str ('large', 'small', 'student') | Dict[str, Callable] |
| `load_large_model()` | 加载 large 模型（BGE-large-zh） | None | SentenceTransformer |
| `load_small_model()` | 加载 small 模型（本地优先） | None | SentenceTransformer |
| `load_student_model()` | 加载 student 模型（本地优先） | None | SentenceTransformer |
| `embed_texts_large/small/student()` | 批量文本转向量 | `texts`: List[str], `normalize`: bool | np.ndarray |
| `embed_query_large/small/student()` | 单条 query 转向量 | `text`: str, `normalize`: bool | np.ndarray |

**支持的模型**:
- **large**: BAAI/bge-large-zh-v1.5（约 1.2GB，1024 维向量，默认）
- **small**: sentence-transformers/all-MiniLM-L6-v2（约 80MB，384 维向量）
- **student**: sentence-transformers/all-MiniLM-L6-v2（约 80MB，384 维向量）

---

### src/embedding/vectorstore.py - 向量数据库模块

**六种组合方案** (chunk_method × model_type):
| 分块方法 | 模型类型 | 向量维度 | 特点 |
|----------|----------|----------|------|
| semantic | large | 1024 | 默认推荐，语义分块 + 高质量检索 |
| semantic | small | 384 | 语义分块 + 平衡性能 |
| semantic | student | 384 | 语义分块 + 快速推理 |
| sliding_window | large | 1024 | 滑动窗口 + 高质量检索 |
| sliding_window | small | 384 | 滑动窗口 + 平衡性能 |
| sliding_window | student | 384 | 滑动窗口 + 快速推理 |

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `build_vectorstore()` | 构建单套向量库 | `chunks`: List[Dict]<br>`chunk_method`: str<br>`model_type`: str | Dict |
| `build_all_vectorstores()` | 构建全部 6 套向量库 | `chunks_semantic`, `chunks_sliding` | Dict[str, Dict] |
| `load_vectorstore()` | 加载单套向量库 | `chunk_method`: str<br>`model_type`: str | Dict |
| `load_all_vectorstores()` | 加载全部 6 套向量库 | - | Dict[str, Dict] |
| `search()` | 检索相似 chunk | `query`: str<br>`vectorstore`: Dict<br>`top_k`: int | List[Tuple] |
| `search_from_config()` | 便捷检索 | `query`: str<br>`chunk_method`: str<br>`model_type`: str | List[Tuple] |
| `get_searcher()` | 获取检索器接口 | `chunk_method`: str<br>`model_type`: str | Dict |

**向量库文件命名**: `embeddings_{chunk_method}_{model_type}.npy`

---

### src/retrieval/retriever.py - 检索模块

**六种组合方案**: semantic/sliding_window × large/small/student

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `retrieve_top_k()` | 检索 Top-K 文档 | `query`: str, `top_k`: int<br>`chunk_method`: str, `model_type`: str | List[Dict] |
| `retrieve_quick()` | 便捷检索 | `query`: str<br>`chunk_method`: str, `model_type`: str | List[Dict] |
| `format_retrieved_context()` | 格式化为上下文 | `retrieved_docs`: List[Dict] | str |
| `get_retriever()` | 获取检索器接口 | `chunk_method`: str, `model_type`: str | Dict |
| `get_all_retrievers()` | 获取全部 6 套检索器 | - | Dict[str, Dict] |

**默认配置**: `chunk_method='semantic'`, `model_type='large'`

---

### src/generation/generator.py - 答案生成模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `generate_answer()` | 根据 Prompt 生成回答 | `prompt`: str | str |

**支持的后端**:
- `dummy`: 测试模式
- `aliyun`: 阿里云 DashScope API（通义千问）

---

### src/generation/prompt_builder.py - 提示词构建模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `build_rag_prompt()` | 构造 RAG 专属 Prompt | `query`: str<br>`retrieved_context`: str | str |

---

### src/pipeline/ingest_pipeline.py - 入库流水线模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `run_ingestion_pipeline()` | 执行完整入库流水线 | `config_path`: str | int |

---

### src/pipeline/rag_pipeline.py - RAG 查询流水线模块

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `run_rag_pipeline()` | 运行完整 RAG 查询流水线 | `query`: str<br>`config_path`: str<br>`model_type`: str | Dict |

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

### src/Interactive interface/ - Web 交互界面模块

**作用**: 提供基于 Flask 的 Web 交互式 RAG 对话界面。

#### app.py - Web 应用后端

| 函数/路由 | 作用 | 输入 | 输出 |
|-----------|------|------|------|
| `RAGSystem` 类 | RAG 系统封装类 | None | 实例对象 |
| `RAGSystem.generate_answer()` | 生成 RAG 回答 | `question`: str | str |
| `/api/rag` (POST) | RAG 请求 API 端点 | `{question: str}` (JSON) | `{answer: str}` (JSON) |
| `/` (GET) | 提供前端页面 | None | HTML |
| `/script.js` (GET) | 提供 JavaScript 文件 | None | JavaScript |

**依赖**:
- `flask>=2.0.0`: Web 框架
- `flask-cors>=3.0.0`: CORS 支持

#### index.html - 前端界面

**作用**: 提供 RAG 对话的用户界面。

**技术栈**:
- Tailwind CSS (CDN): 样式框架
- Font Awesome: 图标库

**主要组件**:
- 对话历史区域：显示用户和 AI 的对话
- 输入区域：用户输入问题
- 加载状态：显示 AI 思考动画

#### script.js - 前端交互逻辑

| 函数 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `addMessageToHistory()` | 添加消息到对话历史 | `sender`: str, `message`: str | DOM 元素 |
| `addLoadingMessage()` | 显示加载动画 | None | 元素 ID |
| `removeLoadingMessage()` | 移除加载动画 | `id`: str | None |

**功能**:
- 监听表单提交
- 发送 POST 请求到 `/api/rag`
- 动态更新对话历史
- 自动滚动到底部

#### run.bat - Windows 启动脚本

**作用**: 一键启动 Flask 服务器并打开浏览器。

**功能**:
- 检查 Python 环境
- 自动安装 Flask 依赖
- 启动 Flask 服务器
- 自动打开浏览器

---

## 四、数据流向图

```
入库流程（两步执行）:

步骤 1 - 分块到 processed:
原始文档 (data/raw/*.docx)
        ↓
[scripts/chunk_to_processed.py]
        ↓
┌───────────────────────────────────┐
│ loading: load_documents_from_dir()│
│        ↓                          │
│ chunking: 两种分块方法             │
│   - semantic (分隔符分块)          │
│   - sliding_window (固定大小)      │
│        ↓                          │
│ preprocessing: preprocess_chunks() │
│   - 全角转半角                     │
│   - 标点标准化                     │
│   - 日期标准化                     │
│   - 去除短块                       │
└───────────────────────────────────┘
        ↓
JSON 文件 (data/processed/):
  - chunks_semantic_cleaned.json
  - chunks_sliding_cleaned.json

步骤 2 - 向量化入库:
JSON 文件 (data/processed/*_cleaned.json)
        ↓
[scripts/embed_to_vectordb.py]
        ↓
┌───────────────────────────────────┐
│ embedding: 6 种组合可选             │
│   chunk_method: semantic/sliding  │
│   model_type: large/small/student │
│        ↓                          │
│ vectorstore: build_vectorstore()  │
└───────────────────────────────────┘
        ↓
向量数据库 (vectordb/):
  - embeddings_semantic_large.npy
  - embeddings_semantic_small.npy
  - embeddings_semantic_student.npy
  - embeddings_sliding_large.npy
  - embeddings_sliding_small.npy
  - embeddings_sliding_student.npy

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
[generation/generator.py] generate_answer() → 阿里云 API
        ↓
答案 + 来源

问题集生成流程:
原始问题文件 (seed_questions.txt)
        ↓
[src/query/clean.py] clean_questions()
        ↓
干净种子问题 (data/query/seed_questions_cleaned.txt)
        ↓
[src/query/generate_dataset.py] generate_dataset()
        ↓
调用 API 批量生成 → 去重清洗 → 生成三个规模数据集
        ↓
data/query/tax_queries_small.txt (200 条)
data/query/tax_queries_medium.txt (500 条)
data/query/tax_queries_large.txt (1000 条)
```

---

## 五、快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型
```bash
python scripts/download_models.py
```

### 3. 配置 API Key
编辑 `configs/configs_local.yaml`:
```yaml
models:
  aliyun_dashscope_api_key: "sk-xxx"
api:
  generation_api_key: "sk-xxx"
```

### 4. 生成问题数据集（可选）
```bash
# 清洗原始问题文件
python src/query/clean.py

# 批量生成问题集
python src/query/generate_dataset.py
```

### 5. 向量化知识库

```bash
# 步骤 1: 分块（两种方法，输出到 data/processed）
python scripts/chunk_to_processed.py

# 步骤 2: 向量化入库（每次一个模型）
python scripts/embed_to_vectordb.py --model large --chunk-method semantic
python scripts/embed_to_vectordb.py --model small --chunk-method semantic
python scripts/embed_to_vectordb.py --model student --chunk-method semantic
```

### 6. 运行查询

**命令行方式**:
```bash
python src/app/main.py
```

**Web 交互界面方式**:
```bash
# 安装 Web 依赖
pip install flask flask-cors

# 启动 Web 服务器
cd src/Interactive interface
python app.py

# 或直接运行启动脚本（Windows）
run.bat

# 在浏览器中访问 http://127.0.0.1:5000
```
