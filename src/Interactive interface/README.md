# RAG 交互界面

## 项目概述

这是一个基于检索增强生成（RAG）技术的交互界面，用于税法问答系统。系统提供：

- **模型选择**：可选择不同的 embedding/retrieval 模型
- **Reranker 开关**：可选择是否启用重排序功能
- **实时问答**：基于真实向量库和生成模型的问答服务

## 目录结构

```
Interactive interface/
├── app.py             # Flask 后端 API
├── index.html         # 前端 HTML 界面
├── script.js          # 前端 JavaScript 逻辑
├── run.bat            # Windows 启动脚本
└── README.md          # 本说明文档
```

## 功能说明

### 界面功能

| 功能 | 说明 |
|------|------|
| 模型选择 | 下拉框选择 embedder/retriever 模型 |
| Chunk 方法 | 根据模型自动显示可选的 chunk 方法（semantic/sliding） |
| Reranker 开关 | 按钮控制是否启用重排序 |
| 配置应用 | 加载选定模型和向量库 |
| 对话交互 | 输入问题，获取 RAG 回答 |

### 可用模型

系统自动扫描 `models/` 目录，支持以下模型：

- `BAAI--bge-large-zh-v1.5` (基座大模型)
- `sentence-transformers--all-MiniLM-L6-v2` (基座小模型)
- `sentence-transformers--all-MiniLM-L6-v2-FFT-*` (全参数微调)
- `sentence-transformers--all-MiniLM-L6-v2-LoRA-*` (LoRA 微调)

## API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/models` | GET | 获取可用模型列表 |
| `/api/config` | POST | 设置模型配置 |
| `/api/config` | GET | 获取当前配置状态 |
| `/api/rag` | POST | 处理查询请求 |
| `/` | GET | 返回前端页面 |
| `/script.js` | GET | 返回 JavaScript 文件 |

## 安装步骤

### 1. 安装依赖

```bash
pip install flask flask-cors
pip install sentence-transformers torch
pip install pyyaml
```

### 2. 准备向量库

确保向量库已构建（由 `embed_to_vectordb.py` 生成）：

```bash
# 构建所有模型的向量库
python scripts/embed_to_vectordb.py

# 或构建指定模型
python scripts/embed_to_vectordb.py --models sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3
```

### 3. 配置生成器后端

在 `configs/configs.yaml` 中设置：

```yaml
models:
  generator_backend: "aliyun"  # 或 "huggingface" / "ollama"
  generator_model_name: "qwen-max"
```

## 运行方式

### Windows

```bash
cd "src/Interactive interface"
run.bat
```

### 或手动启动

```bash
cd "src/Interactive interface"
python app.py
```

### 访问界面

打开浏览器访问：`http://localhost:5000`

## 使用流程

1. **选择模型**：在下拉框中选择要使用的 embedding 模型
2. **选择 Chunk 方法**：根据模型自动或手动选择 chunk 方法
3. **设置 Reranker**：点击开关按钮启用/关闭重排序
4. **点击应用配置**：系统加载向量库和模型
5. **输入问题**：在输入框输入税法相关问题
6. **获取回答**：系统返回 RAG 回答和来源信息

## 返回结果说明

每次查询返回：

```json
{
  "success": true,
  "answer": "回答文本...",
  "sources": ["来源文件1", "来源文件2"],
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "content": "...",
      "retrieval_score": 0.85,
      "reranker_score": 9.2,
      "source": "..."
    }
  ],
  "efficiency": {
    "retrieval_ms": 21.5,
    "rerank_ms": 150.2,
    "total_ms": 171.7
  }
}
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_K_RETRIEVAL` | 20 | 检索阶段返回的 chunk 数量 |
| `TOP_K_FINAL` | 5 | 最终展示给用户的 chunk 数量 |
| `DEFAULT_RERANKER` | BAAI/bge-reranker-v2-gemma | 默认重排序模型 |

## 代码架构

```
app.py
├── RAGSystem 类
│   ├── __init__()           # 初始化，扫描可用模型
│   ├── get_model_list()     # 获取模型列表
│   ├── set_config()         # 设置配置，加载向量库
│   └── query()              # 处理查询
└── Flask 路由
    ├── /api/models          # 模型列表 API
    ├── /api/config          # 配置 API
    ├── /api/rag             # 查询 API
    └── /                    # 前端页面
```

## 注意事项

1. **向量库必须存在**：选择模型前，需确保对应向量库已构建
2. **显存要求**：加载大型模型需要足够的 GPU 显存
3. **API Key**：生成器后端需要配置相应的 API Key（在 `configs_local.yaml`）
4. **端口冲突**：默认端口 5000，如有冲突可在 `app.py` 底部修改

## 问题排查

| 问题 | 解决方案 |
|------|----------|
| 模型列表为空 | 检查 `models/` 目录是否有模型文件 |
| 配置失败：向量库不存在 | 运行 `embed_to_vectordb.py` 构建向量库 |
| 生成失败 | 检查 `configs/configs.yaml` 和 API Key 配置 |
| 端口拒绝连接 | 确认 Flask 服务已启动，检查端口 |

## 相关脚本

- `scripts/embed_to_vectordb.py` - 构建向量库
- `scripts/evaluation_results.py` - 批量评估
- `src/evaluations/run_evaluation.py` - 评估核心模块