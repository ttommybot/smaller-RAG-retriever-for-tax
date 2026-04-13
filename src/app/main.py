# -*- coding: utf-8 -*-
"""
应用主入口模块

提供交互式查询界面，运行完整的 RAG 查询流水线。
"""

from typing import Dict, Optional
import sys
from pathlib import Path
import io

# 设置标准输入使用 UTF-8 编码（防止 Windows 控制台 GBK 编码问题）
if sys.platform == 'win32' and hasattr(sys.stdin, 'buffer'):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 添加项目根目录到导入路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 显式导入模块而不是 from 导入
import src.pipeline.rag_pipeline as rag_pipeline
import src.utils.config_loader as config_loader


def main():
    """Main entry point for RAG application"""
    config_path = "configs/configs.yaml"

    # Load configuration
    config = config_loader.load_config(config_path)

    print("=" * 50)
    print("RAG Tax Query System")
    print("=" * 50)
    print(f"Project: {config['project_name']}")
    print(f"Show sources: {config['app']['show_sources']}")
    print(f"Top-k retrieval: {config['retrieval']['top_k']}")
    print(f"Generator backend: {config['models']['generator_backend']}")
    print(f"Model type: small")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'q' to exit\n")

    # Interactive query loop
    while True:
        try:
            # Get user input
            query = input("Enter your question: ").strip()

            # Check for exit command
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting RAG system. Goodbye!")
                break

            # Run RAG pipeline
            print("\nProcessing your query...")
            result = rag_pipeline.run_rag_pipeline(query, config_path, model_type="small")

            # Display answer
            print(f"\nAnswer:\n{result['answer']}")

            # Display sources if enabled
            if config['app']['show_sources'] and result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            import traceback
            print(f"\nError: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
