# RAG 模型评估对比报告

**生成时间**：2026-04-27 01:33:03

---

## 一、完整评估结果

### 质量指标

| Model | Chunk | Reranker | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|-------|-------|----------|----------|-----------|--------|----------|
| BAAI--bge-large-zh-v1.5 | semantic | False | 0.0000 | 0.5090 | 0.7412 | 0.5203 |
| BAAI--bge-large-zh-v1.5 | semantic | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| BAAI--bge-large-zh-v1.5 | sliding | False | 0.0000 | 0.5260 | 0.7973 | 0.5456 |
| BAAI--bge-large-zh-v1.5 | sliding | True | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| sentence-transformers--all-MiniLM-L6-v2 | semantic | False | 0.0000 | 0.0090 | 0.0333 | 0.0106 |
| sentence-transformers--all-MiniLM-L6-v2 | semantic | True | 0.0000 | 0.0140 | 0.0572 | 0.0175 |
| sentence-transformers--all-MiniLM-L6-v2 | sliding | False | 0.0000 | 0.0070 | 0.0201 | 0.0071 |
| sentence-transformers--all-MiniLM-L6-v2 | sliding | True | 0.0000 | 0.0120 | 0.0274 | 0.0115 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | semantic | False | 0.0000 | 0.0020 | 0.0200 | 0.0044 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | semantic | True | 0.0000 | 0.0050 | 0.0089 | 0.0040 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | semantic | False | 0.0000 | 0.0080 | 0.0269 | 0.0086 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | semantic | True | 0.0000 | 0.0110 | 0.0667 | 0.0169 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | semantic | False | 0.0000 | 0.0090 | 0.0314 | 0.0110 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | semantic | True | 0.0000 | 0.0130 | 0.0429 | 0.0156 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | sliding | False | 0.0000 | 0.0170 | 0.0797 | 0.0234 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | sliding | True | 0.0000 | 0.0210 | 0.0753 | 0.0259 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | sliding | False | 0.0000 | 0.0080 | 0.0211 | 0.0081 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | sliding | True | 0.0000 | 0.0130 | 0.0477 | 0.0157 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | sliding | False | 0.0000 | 0.0110 | 0.0321 | 0.0120 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | sliding | True | 0.0000 | 0.0160 | 0.0587 | 0.0186 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | semantic | False | 0.0000 | 0.0050 | 0.0225 | 0.0064 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | semantic | True | 0.0000 | 0.0060 | 0.0217 | 0.0067 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | semantic | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | semantic | True | 0.0000 | 0.0040 | 0.0147 | 0.0045 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | semantic | False | 0.0000 | 0.0020 | 0.0058 | 0.0020 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | semantic | True | 0.0000 | 0.0060 | 0.0201 | 0.0065 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | sliding | False | 0.0000 | 0.0120 | 0.0619 | 0.0167 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | sliding | True | 0.0000 | 0.0150 | 0.0723 | 0.0201 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | sliding | False | 0.0000 | 0.0080 | 0.0122 | 0.0061 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | sliding | True | 0.0000 | 0.0130 | 0.0587 | 0.0175 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | sliding | False | 0.0000 | 0.0110 | 0.0266 | 0.0100 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | sliding | True | 0.0000 | 0.0150 | 0.0605 | 0.0181 |

### 效率指标

| Model | Chunk | Reranker | 检索(ms) | 重排(ms) | 端到端(ms) | 显存(MB) |
|-------|-------|----------|----------|----------|------------|----------|
| BAAI--bge-large-zh-v1.5 | semantic | False | 22.14 | 0.00 | 22.14 | 1254.18 |
| BAAI--bge-large-zh-v1.5 | semantic | True | 21.66 | 958.61 | 980.27 | 19034.40 |
| BAAI--bge-large-zh-v1.5 | sliding | False | 17.90 | 0.00 | 17.90 | 10814.35 |
| BAAI--bge-large-zh-v1.5 | sliding | True | 18.91 | 2017.14 | 2036.05 | 12450.75 |
| sentence-transformers--all-MiniLM-L6-v2 | semantic | False | 8.23 | 0.00 | 8.23 | 10898.66 |
| sentence-transformers--all-MiniLM-L6-v2 | semantic | True | 8.55 | 513.22 | 521.78 | 11572.01 |
| sentence-transformers--all-MiniLM-L6-v2 | sliding | False | 6.41 | 0.00 | 6.41 | 10898.66 |
| sentence-transformers--all-MiniLM-L6-v2 | sliding | True | 7.59 | 1713.89 | 1721.48 | 12541.67 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | semantic | False | 7.79 | 0.00 | 7.79 | 10985.31 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | semantic | True | 8.72 | 808.77 | 817.49 | 13767.02 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | semantic | False | 7.56 | 0.00 | 7.56 | 11071.96 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | semantic | True | 8.91 | 887.06 | 895.98 | 13220.57 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | semantic | False | 7.82 | 0.00 | 7.82 | 11158.61 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | semantic | True | 8.72 | 958.05 | 966.77 | 14116.43 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | sliding | False | 6.26 | 0.00 | 6.26 | 11245.26 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | sliding | True | 7.49 | 2029.74 | 2037.23 | 12904.39 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | sliding | False | 6.40 | 0.00 | 6.40 | 11331.92 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | sliding | True | 7.45 | 2037.07 | 2044.52 | 13097.06 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | sliding | False | 6.38 | 0.00 | 6.38 | 11418.57 |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | sliding | True | 7.51 | 2156.21 | 2163.72 | 13086.33 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | semantic | False | 7.67 | 0.00 | 7.67 | 11505.22 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | semantic | True | 8.97 | 601.42 | 610.39 | 12387.00 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | semantic | False | 7.63 | 0.00 | 7.63 | 11591.87 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | semantic | True | 8.87 | 706.15 | 715.02 | 12733.57 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | semantic | False | 7.57 | 0.00 | 7.57 | 11678.52 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | semantic | True | 8.48 | 782.35 | 790.83 | 13109.49 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | sliding | False | 6.82 | 0.00 | 6.82 | 11765.17 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | sliding | True | 7.60 | 2033.69 | 2041.29 | 13408.75 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | sliding | False | 6.39 | 0.00 | 6.39 | 11851.83 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | sliding | True | 7.66 | 1978.05 | 1985.71 | 13515.03 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | sliding | False | 6.48 | 0.00 | 6.48 | 11938.48 |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | sliding | True | 7.72 | 1876.79 | 1884.51 | 13503.84 |

---

## 二、对比分析

### 1. 微调是否有效？

对比：MiniLM vs 微调后 MiniLM（有无 reranker）


#### SEMANTIC 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比基线提升 |
|------|----------|----------|-----------|---------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | False | 0.0000 | 0.0020 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | False | 0.0000 | 0.0090 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | False | 0.0000 | 0.0050 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | False | 0.0000 | 0.0000 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | False | 0.0000 | 0.0020 | +0.00% |
| MiniLM (基线) | False | 0.0000 | 0.0090 | -- |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | True | 0.0000 | 0.0050 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | True | 0.0000 | 0.0110 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | True | 0.0000 | 0.0060 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | True | 0.0000 | 0.0040 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | True | 0.0000 | 0.0060 | +0.00% |
| MiniLM (基线) | True | 0.0000 | 0.0140 | -- |

#### SLIDING 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比基线提升 |
|------|----------|----------|-----------|---------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | False | 0.0000 | 0.0170 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | False | 0.0000 | 0.0110 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | False | 0.0000 | 0.0120 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | False | 0.0000 | 0.0110 | +0.00% |
| MiniLM (基线) | False | 0.0000 | 0.0070 | -- |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | True | 0.0000 | 0.0210 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | True | 0.0000 | 0.0160 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | True | 0.0000 | 0.0150 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | True | 0.0000 | 0.0150 | +0.00% |
| MiniLM (基线) | True | 0.0000 | 0.0120 | -- |

### 2. 微调后小模型能否替代大模型？

对比：微调后 MiniLM vs BGE-Large（有无 reranker）


#### SEMANTIC 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large |
|------|----------|----------|-----------|----------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | False | 0.0000 | 0.0020 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | False | 0.0000 | 0.0090 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | False | 0.0000 | 0.0050 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | False | 0.0000 | 0.0000 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | False | 0.0000 | 0.0020 | +0.00% |
| BGE-Large (基线) | False | 0.0000 | 0.5090 | -- |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | True | 0.0000 | 0.0050 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | True | 0.0000 | 0.0110 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | True | 0.0000 | 0.0060 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | True | 0.0000 | 0.0040 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | True | 0.0000 | 0.0060 | +0.00% |
| BGE-Large (基线) | True | 0.0000 | 1.0000 | -- |

#### SLIDING 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large |
|------|----------|----------|-----------|----------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | False | 0.0000 | 0.0170 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | False | 0.0000 | 0.0110 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | False | 0.0000 | 0.0120 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | False | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | False | 0.0000 | 0.0110 | +0.00% |
| BGE-Large (基线) | False | 0.0000 | 0.5260 | -- |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | True | 0.0000 | 0.0210 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | True | 0.0000 | 0.0160 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | True | 0.0000 | 0.0150 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | True | 0.0000 | 0.0130 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | True | 0.0000 | 0.0150 | +0.00% |
| BGE-Large (基线) | True | 0.0000 | 1.0000 | -- |

### 3. 微调后小模型能否接近大模型+reranker？

对比：微调后 MiniLM（无 reranker）vs BGE-Large + Reranker


#### SEMANTIC 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large+Reranker |
|------|----------|----------|-----------|------------------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 | No | 0.0000 | 0.0020 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4 | No | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5 | No | 0.0000 | 0.0090 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 | No | 0.0000 | 0.0050 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 | No | 0.0000 | 0.0000 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 | No | 0.0000 | 0.0020 | +0.00% |
| BGE-Large | Yes (基准) | 0.0000 | 1.0000 | -- |

#### SLIDING 分块

| 模型 | Reranker | Recall@5 | Recall@10 | 相比 BGE-Large+Reranker |
|------|----------|----------|-----------|------------------------|
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3 | No | 0.0000 | 0.0170 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4 | No | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5 | No | 0.0000 | 0.0110 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 | No | 0.0000 | 0.0120 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 | No | 0.0000 | 0.0080 | +0.00% |
| sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 | No | 0.0000 | 0.0110 | +0.00% |
| BGE-Large | Yes (基准) | 0.0000 | 1.0000 | -- |

---

## 三、结论与建议

根据以上对比分析，可以得出以下结论：

1. **微调有效性**：观察 FFT/LoRA 模型相比 MiniLM 基线的提升幅度
2. **替代可能性**：观察微调后 MiniLM 与 BGE-Large 的差距
3. **效率权衡**：对比显存占用和推理延迟

建议选择综合表现最优的模型配置进行部署。
