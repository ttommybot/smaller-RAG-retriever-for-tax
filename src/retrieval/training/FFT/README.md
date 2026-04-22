# 全参数微调训练（Full Parameter Fine-Tuning）

本目录包含使用 triplet 损失对 sentence-transformers 模型进行全参数微调的核心模块。

## 目录结构

```
FFT/
├── training_FFT.py   # 核心训练模块（被脚本调用）
├── README.md         # 说明文档
└── config.yaml       # 训练配置（可选）
```

## 训练流程

### 固定超参数
| 超参数 | 值 | 说明 |
|--------|-----|------|
| Batch size | 64 | 每批次样本数 |
| Warmup steps | 10% | 预热步数为总步数的 10% |
| Loss function | TripletLoss | 三元组损失 |

### 自动确定
| 超参数 | 方法 | 说明 |
|--------|------|------|
| 学习率 | 学习率查找器 | 训练 100 个 batch，找损失下降最快的点 |
| 训练轮数 | 早停法 | 验证损失连续 3 轮不改善则停止 |

### 手动指定
| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| Margin | 0.5 | Triplet Loss 边距，推荐 0.3 ~ 0.7 |

## 使用方法

### 方式一：使用训练脚本（推荐）

```bash
# 使用 semantic 分块数据，margin=0.5
python scripts/train_minilm_FFT.py --chunk-method semantic --margin 0.5

# 使用 sliding_window 分块数据，margin=0.3
python scripts/train_minilm_FFT.py --chunk-method sliding_window --margin 0.3

# 手动指定学习率，跳过学习率查找器
python scripts/train_minilm_FFT.py --chunk-method semantic --margin 0.5 --manual-lr 2e-5

# 跳过学习率查找器，使用默认学习率
python scripts/train_minilm_FFT.py --chunk-method semantic --margin 0.5 --skip-lr-finder
```

### 方式二：直接调用核心模块

```python
from retrieval.training.FFT.training_FFT import load_training_data, train_fft

# 加载训练数据
train_data = load_training_data(
    data_file="semantic_trained.json",
    chunk_method="semantic"
)

# 训练
train_fft(
    train_examples=train_data,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_name="sentence-transformers/all-MiniLM-L6-v2-FFT",
    batch_size=64,
    learning_rate=2e-5,
    num_epochs=5,
    margin=0.5,
    warmup_ratio=0.1
)
```

## 训练数据格式

### 输入文件（data/training/）

```jsonl
{"query": "什么是增值税？", "positive_chunk_id": "法律_会计法_chunk_0001", "negative_chunk_id": "法规_税法_chunk_0023"}
{"query": "如何计算个税？", "positive_chunk_id": "税法_个税法_chunk_0005", "negative_chunk_id": "法规_增值税_chunk_0012"}
```

### 自动推断
| chunk_method | data_file | chunks_file |
|--------------|-----------|-------------|
| semantic | semantic_trained.json | chunks_semantic_cleaned.json |
| sliding_window | sliding_trained.json | chunks_sliding_cleaned.json |

## 输出

训练完成后，模型保存到：
```
models/sentence-transformers--all-MiniLM-L6-v2-FFT/
```

学习率查找结果保存到：
```
data/training/lr_finder_results.json
```

## 参数说明（scripts/train_minilm_FFT.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk-method` | semantic | 分块方法 |
| `--margin` | 0.5 | Triplet Loss 边距 |
| `--data-file` | 自动 | 训练数据文件名 |
| `--chunks-file` | 自动 | Chunks 文件名 |
| `--model-name` | sentence-transformers/all-MiniLM-L6-v2 | 基座模型 |
| `--output-name` | sentence-transformers/all-MiniLM-L6-v2-FFT | 输出模型 |
| `--max-epochs` | 10 | 最大训练轮数 |
| `--patience` | 3 | 早停容忍轮数 |
| `--no-cuda` | False | 不使用 GPU |
| `--skip-lr-finder` | False | 跳过学习率查找 |
| `--manual-lr` | None | 手动指定学习率 |

## Margin 选择建议

| 数据特点 | 推荐 margin |
|----------|-------------|
| 正负例差异明显（简单） | 0.3 |
| 正负例有一定差异（中等） | 0.5 |
| 正负例很相似（困难） | 0.7 |

**调参建议**：从 0.5 开始，对比 0.3 和 0.7 的效果，选验证集损失最低的。

## 加载微调后的模型

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/sentence-transformers--all-MiniLM-L6-v2-FFT")
embeddings = model.encode(["查询文本"])
```
