# -*- coding: utf-8 -*-
"""
RAG 交互界面后端

功能：
1. 可选择模型作为 embedder & retriever
2. 可选择是否使用 reranker
3. 实际调用项目中的检索和生成模块
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# 获取当前目录（Interactive interface）
CURRENT_DIR = Path(__file__).parent

# 获取项目根目录
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 切换工作目录到项目根目录（解决相对路径问题）
os.chdir(PROJECT_ROOT)

# 添加 src 到 Python 路径
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding.vectorstore import (
    load_vectorstore_for_custom_model,
    search_custom_vectorstore,
    get_vectorstore_model_dir,
    get_chunk_methods_for_model,
)
from embedding.embedder import get_custom_embedder
from reranking.reranker import load_reranker, rerank_chunks
from generation.generator import generate_answer
from generation.prompt_builder import build_rag_prompt

app = Flask(__name__, static_folder=str(CURRENT_DIR))
CORS(app)

# ==========================================
# 配置
# ==========================================
MODELS_DIR = PROJECT_ROOT / "models"
EXCLUDED_MODELS = ["BAAI--bge-reranker-v2-gemma"]
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-gemma"
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5


# ==========================================
# 获取可用模型列表
# ==========================================
def get_available_models() -> List[str]:
    """获取 models/ 目录下所有可用模型。"""
    models = []
    if not MODELS_DIR.exists():
        return models

    for item in MODELS_DIR.iterdir():
        if item.is_dir() and item.name != ".gitkeep":
            if item.name in EXCLUDED_MODELS:
                continue
            models.append(item.name)

    return sorted(models)


# ==========================================
# RAG 系统类
# ==========================================
class RAGSystem:
    def __init__(self):
        """初始化 RAG 系统。"""
        self.model_name: Optional[str] = None
        self.chunk_method: str = "semantic"
        self.use_reranker: bool = False
        self.vectorstore: Optional[Dict[str, Any]] = None
        self.embedder_model: Optional[Any] = None
        self.reranker: Optional[Any] = None
        self.initialized: bool = False

        # 可用模型列表
        self.available_models = get_available_models()

        print(f"RAG 系统初始化完成")
        print(f"可用模型：{len(self.available_models)} 个")

    def get_model_list(self) -> List[Dict[str, Any]]:
        """获取模型列表（带 chunk 方法信息）。"""
        model_list = []
        for model_name in self.available_models:
            chunk_methods = get_chunk_methods_for_model(model_name)
            model_list.append({
                "name": model_name,
                "chunk_methods": chunk_methods
            })
        return model_list

    def set_config(
        self,
        model_name: str,
        chunk_method: str = "semantic",
        use_reranker: bool = False
    ) -> Dict[str, Any]:
        """
        设置 RAG 配置。

        返回
        -------
        Dict[str, Any]
            配置结果信息。
        """
        result = {
            "success": False,
            "model_name": model_name,
            "chunk_method": chunk_method,
            "use_reranker": use_reranker,
            "error": None
        }

        try:
            # 检查模型是否存在
            if model_name not in self.available_models:
                result["error"] = f"模型不存在：{model_name}"
                return result

            # 检查向量库是否存在
            vectorstore_dir = get_vectorstore_model_dir(model_name, chunk_method)
            embeddings_path = vectorstore_dir / "embeddings.npy"

            if not embeddings_path.exists():
                result["error"] = f"向量库不存在：{vectorstore_dir}"
                return result

            # 加载向量库
            print(f"加载向量库：{model_name} ({chunk_method})")
            self.vectorstore = load_vectorstore_for_custom_model(model_name, chunk_method)

            # 加载 embedder
            embedder = get_custom_embedder(model_name)
            self.embedder_model = embedder['model']

            # 加载 reranker（如果启用）
            if use_reranker:
                print(f"加载 Reranker：{DEFAULT_RERANKER}")
                self.reranker = load_reranker(DEFAULT_RERANKER)
            else:
                self.reranker = None

            # 更新配置
            self.model_name = model_name
            self.chunk_method = chunk_method
            self.use_reranker = use_reranker
            self.initialized = True

            result["success"] = True
            result["num_chunks"] = len(self.vectorstore['chunks'])
            result["embedding_dim"] = self.vectorstore['embedding_dim']

            print(f"配置成功：{model_name} | {chunk_method} | reranker={use_reranker}")

        except Exception as e:
            result["error"] = str(e)
            print(f"配置失败：{e}")

        return result

    def query(self, question: str) -> Dict[str, Any]:
        """
        处理查询请求。

        返回
        -------
        Dict[str, Any]
            包含回答和来源的结果。
        """
        result = {
            "success": False,
            "answer": "",
            "sources": [],
            "retrieved_chunks": [],
            "efficiency": {},
            "error": None
        }

        if not self.initialized:
            result["error"] = "系统未初始化，请先选择模型"
            return result

        try:
            # 1. 检索
            retrieval_start = time.time()
            retrieved_results = search_custom_vectorstore(
                query=question,
                vectorstore=self.vectorstore,
                model=self.embedder_model,
                top_k=TOP_K_RETRIEVAL
            )
            retrieval_end = time.time()
            retrieval_ms = (retrieval_end - retrieval_start) * 1000

            # 构建候选 chunks
            candidate_chunks = []
            for chunk, score in retrieved_results:
                candidate_chunks.append({
                    "chunk_id": chunk.get("id", ""),
                    "content": chunk.get("content", ""),
                    "retrieval_score": float(score),
                    "source": chunk.get("metadata", {}).get("file_name", "未知来源")
                })

            # 2. 重排（如果启用）
            rerank_ms = 0.0
            if self.use_reranker and self.reranker and len(candidate_chunks) > 0:
                rerank_start = time.time()
                reranked = rerank_chunks(
                    query=question,
                    chunks=candidate_chunks,
                    top_k=TOP_K_FINAL,
                    text_key="content"
                )
                rerank_end = time.time()
                rerank_ms = (rerank_end - rerank_start) * 1000

                # 构建最终 chunks
                final_chunks = []
                for item in reranked:
                    raw = item.get("raw", {})
                    final_chunks.append({
                        "chunk_id": raw.get("chunk_id", ""),
                        "content": item.get("text", ""),
                        "retrieval_score": raw.get("retrieval_score", 0),
                        "reranker_score": item.get("score", 0),
                        "source": raw.get("source", "未知来源")
                    })
            else:
                final_chunks = candidate_chunks[:TOP_K_FINAL]

            # 3. 构建上下文
            context_parts = []
            for i, chunk in enumerate(final_chunks, 1):
                context_parts.append(
                    f"[资料{i} - 来源：{chunk['source']}]\n{chunk['content']}"
                )
            context = "\n\n".join(context_parts)

            # 4. 生成回答
            prompt = build_rag_prompt(question, context)
            answer = generate_answer(prompt)

            # 5. 构建结果
            result["success"] = True
            result["answer"] = answer
            result["sources"] = [chunk["source"] for chunk in final_chunks]
            result["retrieved_chunks"] = final_chunks
            result["efficiency"] = {
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "total_ms": retrieval_ms + rerank_ms
            }
            result["config"] = {
                "model_name": self.model_name,
                "chunk_method": self.chunk_method,
                "use_reranker": self.use_reranker
            }

        except Exception as e:
            result["error"] = str(e)
            print(f"查询失败：{e}")

        return result


# 初始化 RAG 系统
rag_system = RAGSystem()


# ==========================================
# API 端点
# ==========================================

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表。"""
    model_list = rag_system.get_model_list()
    return jsonify({
        "models": model_list,
        "total": len(model_list)
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    """设置 RAG 配置。"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '缺少配置参数'}), 400

    model_name = data.get('model_name')
    chunk_method = data.get('chunk_method', 'semantic')
    use_reranker = data.get('use_reranker', False)

    if not model_name:
        return jsonify({'error': '缺少模型名称'}), 400

    result = rag_system.set_config(
        model_name=model_name,
        chunk_method=chunk_method,
        use_reranker=use_reranker
    )

    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取当前配置。"""
    return jsonify({
        "model_name": rag_system.model_name,
        "chunk_method": rag_system.chunk_method,
        "use_reranker": rag_system.use_reranker,
        "initialized": rag_system.initialized
    })


@app.route('/api/rag', methods=['POST'])
def rag_endpoint():
    """处理 RAG 查询请求。"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': '缺少问题参数'}), 400

    question = data['question']

    # 调用 RAG 系统
    result = rag_system.query(question)

    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


@app.route('/')
def index():
    """提供前端页面。"""
    return send_from_directory(str(CURRENT_DIR), 'index.html')


@app.route('/script.js')
def script():
    """提供 JavaScript 文件。"""
    return send_from_directory(str(CURRENT_DIR), 'script.js', mimetype='text/javascript')


# ==========================================
# 主函数
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("RAG 交互界面")
    print("=" * 60)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"可用模型：{rag_system.available_models}")
    print(f"启动地址：http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)