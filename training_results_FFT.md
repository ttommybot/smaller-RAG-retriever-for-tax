# MiniLM 全参数微调训练结果对比

**生成时间**：2026-04-26 11:07:03

---

## 训练配置

- Batch size: 64
- Warmup: 10%
- Loss: TripletLoss
- 最大轮数：20
- 早停容忍：3 轮

## 结果对比

| Chunk 方法 | Margin | 训练时间(s) | 显存(MB) | 实际轮数 | 最佳验证损失 | 学习率 | 模型路径 |
|------------|--------|-------------|----------|----------|--------------|--------|----------|
| semantic | 0.3 | 355.46 | 6302.81 | 15 | 0.1731 | 7.76e-05 | models/
| semantic | 0.4 | 333.40 | 6478.95 | 14 | 0.2304 | 1.29e-05 | models/
| semantic | 0.5 | 461.85 | 6654.46 | 20 | 0.2933 | 1.48e-05 | models/
| sliding | 0.3 | 340.49 | 6303.39 | 12 | 0.1565 | 3.89e-05 | models/
| sliding | 0.4 | 213.73 | 6478.75 | 7 | 0.2443 | 1.02e-04 | models/
| sliding | 0.5 | 264.44 | 6566.18 | 9 | 0.2868 | 1.10e-04 | models/

---

## 详细结果

### semantic + margin=0.3

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：15
- 最佳验证损失：0.1731
- 训练时间：355.46 秒
- 显存占用：6302.81 MB
- 学习率：7.76e-05
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3

### semantic + margin=0.4

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：14
- 最佳验证损失：0.2304
- 训练时间：333.40 秒
- 显存占用：6478.95 MB
- 学习率：1.29e-05
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.4

### semantic + margin=0.5

- 训练样本：3892
- 验证样本：432
- 实际训练轮数：20
- 最佳验证损失：0.2933
- 训练时间：461.85 秒
- 显存占用：6654.46 MB
- 学习率：1.48e-05
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.5

### sliding + margin=0.3

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：12
- 最佳验证损失：0.1565
- 训练时间：340.49 秒
- 显存占用：6303.39 MB
- 学习率：3.89e-05
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.3

### sliding + margin=0.4

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：7
- 最佳验证损失：0.2443
- 训练时间：213.73 秒
- 显存占用：6478.75 MB
- 学习率：1.02e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.4

### sliding + margin=0.5

- 训练样本：4494
- 验证样本：499
- 实际训练轮数：9
- 最佳验证损失：0.2868
- 训练时间：264.44 秒
- 显存占用：6566.18 MB
- 学习率：1.10e-04
- 模型路径：/root/rag_for_tax/models/sentence-transformers--all-MiniLM-L6-v2-FFT-sliding-0.5

