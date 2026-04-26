#!/bin/bash
# -*- coding: utf-8 -*-
"""
为两版 chunks 分别生成 triplet 数据

用法：
python scripts/run_dual_triplet_generation.py \
    --query-path data/query/tax_queries_large.txt \
    --output-dir data/training \
    --min-margin 0.05 \
    --max-negatives-per-query 3

python scripts/run_dual_triplet_generation.py \
    --query-path data/query/evaluation_queries.txt \
    --output-dir data/evaluations \
    --min-margin 0.05 \
    --max-negatives-per-query 3
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_pipeline_for_version(
    query_path: str,
    chunk_version: str,
    output_dir: str,
    **kwargs
) -> dict:
    """为一个 chunk 版本运行管道。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径
    triplet_output = output_dir / f"{chunk_version}_triplets.json"
    top10_output = output_dir / f"top10_{chunk_version}.jsonl"

    # 自动确定 chunk 文件路径
    chunk_file = Path(__file__).parent.parent / f"data/processed/chunks_{chunk_version}_cleaned.json"
    
    if not chunk_file.exists():
        return {
            "chunk_version": chunk_version,
            "exit_code": 1,
            "triplet_output": str(triplet_output),
            "top10_output": str(top10_output),
            "error": f"Chunk 文件不存在: {chunk_file}"
        }

    # 构建命令
    cmd = [
        sys.executable,
        "-m", "src.pipeline.unified_triplet_pipeline",
        "--query-path", query_path,
        "--chunk-version", chunk_version,
        "--chunk-file", str(chunk_file),
        "--triplet-output", str(triplet_output),
        "--top10-output", str(top10_output),
    ]

    # 处理其他参数
    for key, value in kwargs.items():
        if value is None:
            continue
        
        arg_name = key.replace('_', '-')
        
        # 特殊处理：use_chunk_similarity_filter 对应 --no-use-chunk-similarity-filter
        if key == "use_chunk_similarity_filter":
            # 默认是 True，只有 False 时才添加 --no-... 参数
            if not value:
                cmd.append("--no-use-chunk-similarity-filter")
        elif isinstance(value, bool):
            # 其他布尔参数：True 时添加 --flag
            if value:
                cmd.append(f"--{arg_name}")
        else:
            cmd.append(f"--{arg_name}")
            cmd.append(str(value))

    print("\n" + "=" * 80)
    print(f"【开始】{chunk_version.upper()} 版本 Triplet 生成")
    print("=" * 80)
    print(f"Query 文件: {query_path}")
    print(f"Chunk 版本: {chunk_version}")
    print(f"Triplet 输出: {triplet_output}")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 80 + "\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return {
        "chunk_version": chunk_version,
        "exit_code": result.returncode,
        "triplet_output": str(triplet_output),
        "top10_output": str(top10_output),
    }


def main():
    parser = argparse.ArgumentParser(description="为两版 chunks 分别生成 triplet 数据")
    parser.add_argument("--query-path", required=True, help="Query 文件路径")
    parser.add_argument("--output-dir", default="data/training", help="输出目录")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-pos-score", type=float, default=0.8)
    parser.add_argument("--max-neg-score", type=float, default=0.7)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--max-negatives-per-query", type=int, default=1)
    
    # 修改：使用 store_false 匹配原始脚本
    parser.add_argument(
        "--no-use-chunk-similarity-filter",
        action="store_false",
        dest="use_chunk_similarity_filter",
        default=True,
        help="禁用 chunk 相似度过滤"
    )
    parser.add_argument("--min-chunk-diff", type=float, default=0.05)
    parser.add_argument("--max-chunk-similarity", type=float, default=0.95)
    
    # 跳过选项
    parser.add_argument("--skip-semantic", action="store_true", help="跳过 semantic 版本")
    parser.add_argument("--skip-sliding", action="store_true", help="绕过 sliding 版本")
    
    args = parser.parse_args()

    # 验证 query 文件存在
    if not Path(args.query_path).exists():
        print(f"❌ Query 文件不存在: {args.query_path}")
        sys.exit(1)

    print("\n" + "🚀" * 40)
    print("统一 Triplet 生成 - 双版本执行")
    print("🚀" * 40)

    # 确定要运行的版本
    versions = []
    if not args.skip_semantic:
        versions.append("semantic")
    if not args.skip_sliding:
        versions.append("sliding")
    
    results = []

    for version in versions:
        result = run_pipeline_for_version(
            query_path=args.query_path,
            chunk_version=version,
            output_dir=args.output_dir,
            top_k=args.top_k,
            min_pos_score=args.min_pos_score,
            max_neg_score=args.max_neg_score,
            min_margin=args.min_margin,
            max_negatives_per_query=args.max_negatives_per_query,
            use_chunk_similarity_filter=args.use_chunk_similarity_filter,
            min_chunk_diff=args.min_chunk_diff,
            max_chunk_similarity=args.max_chunk_similarity,
        )
        results.append(result)

    # 输出总结
    print("\n" + "=" * 80)
    print("【完成总结】")
    print("=" * 80)
    
    all_success = True
    for result in results:
        status = "成功" if result["exit_code"] == 0 else "失败"
        print(f"\n{status} - {result['chunk_version'].upper()}")
        print(f"   Triplet: {result['triplet_output']}")
        print(f"   Top10:   {result['top10_output']}")
        if result.get("error"):
            print(f"   错误:    {result['error']}")
            all_success = False

    if all_success:
        print("\n" + "🎉" * 40)
        print("所有版本生成完成！")
        print("🎉" * 40)
    else:
        print("\n⚠️ 某些版本生成失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
