# -*- coding: utf-8 -*-
"""
应用主入口模块

提供交互式查询界面，运行完整的 RAG 查询流水线。
"""

from typing import Dict, Optional
import sys
from pathlib import Path

# 添加 src 目录到导入路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pipeline.rag_pipeline import run_rag_pipeline, RAGResult
from utils.config_loader import load_config


def main():
    """Main entry point for RAG application"""
    config_path = "configs/configs.yaml"

    # Load configuration
    config = load_config(config_path)

    print("=" * 50)
    print("RAG Tax Query System")
    print("=" * 50)
    print(f"Project: {config['project_name']}")
    print(f"Show sources: {config['app']['show_sources']}")
    print(f"Top-k retrieval: {config['retrieval']['top_k']}")
    print(f"Generator backend: {config['models']['generator_backend']}")
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
            result: RAGResult = run_rag_pipeline(query, config_path)

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
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
