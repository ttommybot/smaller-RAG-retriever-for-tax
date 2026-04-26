# MiniLM LoRA 微调训练结果对比

**生成时间**：2026-04-26 16:23:31

---

## 训练配置

- Batch size: 64
- Loss: TripletLoss
- LoRA Rank: 16
- 最大轮数：10
- 早停容忍：3 轮

## 结果对比

| Chunk 方法 | Margin | 训练时间(s) | 显存(MB) | 实际轮数 | 最佳验证损失 | 学习率 | 模型路径 |
|------------|--------|-------------|----------|----------|--------------|--------|----------|
| semantic | 0.3 | 177.52 | 4868.69 | 10 | 0.2284 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3 |
| semantic | 0.4 | 177.57 | 4954.74 | 10 | 0.3051 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4 |
| semantic | 0.5 | 178.90 | 4865.04 | 10 | 0.3776 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5 |
| sliding | 0.3 | 213.49 | 4954.74 | 10 | 0.1869 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3 |
| sliding | 0.4 | 213.86 | 5042.90 | 10 | 0.2476 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4 |
| sliding | 0.5 | 213.55 | 5128.52 | 10 | 0.3265 | 2.00e-04 | /root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5 |

---

## 详细结果

### semantic + margin=0.3

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：10
- 最佳验证损失：0.2284
- 训练时间：177.52 秒
- 显存占用：4868.69 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.3

### semantic + margin=0.4

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：10
- 最佳验证损失：0.3051
- 训练时间：177.57 秒
- 显存占用：4954.74 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.4

### semantic + margin=0.5

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：10
- 最佳验证损失：0.3776
- 训练时间：178.90 秒
- 显存占用：4865.04 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-semantic-0.5

### sliding + margin=0.3

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：10
- 最佳验证损失：0.1869
- 训练时间：213.49 秒
- 显存占用：4954.74 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.3

### sliding + margin=0.4

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：10
- 最佳验证损失：0.2476
- 训练时间：213.86 秒
- 显存占用：5042.90 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.4

### sliding + margin=0.5

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：10
- 最佳验证损失：0.3265
- 训练时间：213.55 秒
- 显存占用：5128.52 MB
- 学习率：2.00e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-LoRA-sliding-0.5

