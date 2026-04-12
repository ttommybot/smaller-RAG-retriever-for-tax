# RAG对话系统

## 项目概述

这是一个基于检索增强生成（RAG）技术的对话系统，用于课程项目。系统提供了一个简洁的交互式界面，用户可以输入问题，系统通过RAG技术生成回答。

## 项目结构

```
DDA4210/
├── index.html         # 前端HTML界面
├── script.js          # 前端JavaScript逻辑
├── app.py             # 后端Flask API
└── README.md          # 项目说明文档
```

## 功能说明

- **前端界面**：使用Tailwind CSS构建的现代、响应式对话界面
- **后端API**：基于Flask的RESTful API，处理用户请求并返回RAG回答
- **RAG系统接口**：预留了模型集成位置，可根据实际需求集成不同的模型

## 安装步骤

1. **安装Python依赖**：
   ```bash
   python -m pip install flask flask-cors
   ```

2. **安装可选的模型依赖**（用于实际RAG系统）：
   ```bash
   python -m pip install sentence-transformers transformers faiss-cpu torch
   ```

## 运行方式

1. **启动后端服务器**：
   ```bash
   python app.py
   ```

2. **访问前端界面**：
   打开浏览器，访问 `http://127.0.0.1:5000`

## 代码说明

### 前端代码

- **index.html**：构建了对话界面，包含导航栏、对话历史区域和输入区域
- **script.js**：实现了用户输入处理、发送请求到后端API、显示对话历史和加载状态

### 后端代码

- **app.py**：
  - 实现了Flask API，处理 `/api/rag` 端点的POST请求
  - 定义了 `RAGSystem` 类，预留了模型集成接口
  - 提供了模拟的RAG回答功能，可根据实际需求替换为真实的RAG系统

### RAG系统集成

在 `RAGSystem` 类中，预留了以下模型的集成位置：

1. **嵌入模型**（Embedder）：如 BAAI/bge-large-zh-v1.5 或微调后的 minilm
2. **向量数据库**（Vector Database）：如 FAISS, Chroma, Pinecone 等
3. **重排序模型**（Reranker）：如 BAAI/bge-reranker-v2-gemma
4. **生成模型**（Generator）：如 LLaMA, ChatGLM 等

## 后续扩展建议

1. **集成真实模型**：根据项目需求，集成实际的嵌入模型、向量数据库、重排序模型和生成模型
2. **添加文档处理功能**：支持上传和处理文档，构建向量数据库
3. **优化前端界面**：添加更多交互功能，如历史记录、主题切换等
4. **添加评估指标**：实现实验设计中提到的评估指标，如 Recall@k, Hit Rate@k, MRR@k, nDCG@k 等
5. **性能优化**：优化查询速度和内存使用，提高系统效率

## 注意事项

- 本系统目前使用的是模拟的RAG回答，实际应用中需要替换为真实的RAG系统
- 开发环境中使用的是Flask的开发服务器，生产环境中建议使用WSGI服务器
- 模型的选择和配置应根据实际硬件资源和性能需求进行调整
