# RAG for Tax - Tax Law Retrieval-Augmented Generation System

A tax law QA system based on Retrieval-Augmented Generation (RAG) technology, fine-tuning MiniLM models through knowledge distillation for efficient and accurate tax law retrieval and question answering.

## Highlights

- **Knowledge Distillation**: Using BGE-large + BGE-reranker to generate distillation data, fine-tune MiniLM to improve retrieval capability
- **Dual Fine-tuning Strategies**: Supports Full Parameter Fine-Tuning (FFT) and LoRA fine-tuning
- **Multiple Chunking Methods**: Supports semantic chunking and sliding window chunking
- **Comprehensive Evaluation**: Quality metrics (Recall@k, HitRate@k, MRR@k, nDCG@k) + efficiency metrics (time/memory)
- **Interactive Interface**: Web UI with model selection, reranker toggle, real-time QA

---

## Directory Structure

```
rag_for_tax/
├── configs/                      # Configuration files
│   ├── configs.yaml              # Main configuration
│   ├── configs.yaml.template     # Configuration template
│   └── configs_local.yaml        # Local config (API keys, not in Git)
│
├── models/                       # Local models directory
│   ├── BAAI--bge-large-zh-v1.5/           # BGE large Chinese embedding model
│   ├── BAAI--bge-reranker-v2-gemma/       # BGE reranker model
│   ├── sentence-transformers--all-MiniLM-L6-v2/          # MiniLM base model
│   ├── sentence-transformers--all-MiniLM-L6-v2-FFT-*/    # FFT fine-tuned models (6)
│   └── sentence-transformers--all-MiniLM-L6-v2-LoRA-*/   # LoRA fine-tuned models (6)
│
├── scripts/                      # Execution scripts
│   ├── download_models.py        # Model download
│   ├── chunk_to_processed.py     # Document chunking
│   ├── embed_to_vectordb.py      # Vectorization & storage
│   ├── train_minilm_FFT.py       # FFT training
│   ├── train_minilm_LoRA.py      # LoRA training
│   ├── evaluation_results.py     # Batch evaluation
│   └── addition_ingestion.py     # LoRA model ingestion
│
├── src/                          # Core modules
│   ├── loading/                  # Document loading
│   ├── chunking/                 # Chunking & preprocessing
│   ├── embedding/                # Embedding & vectorstore
│   ├── retrieval/                # Retrieval & training
│   ├── reranking/                # Reranking
│   ├── generation/               # Answer generation
│   ├── evaluations/              # Evaluation module
│   ├── query/                    # Query processing
│   ├── pipeline/                 # Pipeline
│   ├── utils/                    # Utilities
│   ├── app/                      # CLI application
│   └── Interactive interface/    # Web interface
│
├── data/                         # Data directory
│   ├── raw/                      # Raw documents (Word)
│   ├── processed/                # Processed chunks
│   ├── training/                 # Training data
│   ├── evaluations/              # Evaluation data
│   └── query/                    # Query dataset
│
├── vectordb/                     # Vector database
│   ├── BAAI--bge-large-zh-v1.5/
│   ├── sentence-transformers--all-MiniLM-L6-v2/
│   └── ... (all fine-tuned models)
│
├── requirements.txt
├── README.md
└── structure.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install flask flask-cors  # Web interface dependencies
```

### 2. Configure API Keys

Create `configs/configs_local.yaml`:

```yaml
models:
  generator_backend: "aliyun"
  generator_model_name: "qwen-max"
  aliyun_dashscope_api_key: "your_api_key"

api:
  generation_url: "https://api.closeai-asia.com/v1/chat/completions"
  generation_api_key: "your_api_key"
  generation_model: "deepseek-chat"
```

### 3. Download Models

```bash
python scripts/download_models.py
```

### 4. Build Vector Database

```bash
# Step 1: Document chunking
python scripts/chunk_to_processed.py

# Step 2: Vectorization (all models)
python scripts/embed_to_vectordb.py
```

### 5. Start the System

**CLI mode**:
```bash
python src/app/main.py
```

**Web interface**:
```bash
cd "src/Interactive interface"
python app.py
# Visit http://localhost:5000
```

---

## Model Description

### Embedding Models

| Model | Dimension | Size | Description |
|-------|-----------|------|-------------|
| BAAI/bge-large-zh-v1.5 | 1024 | ~1.2GB | Large Chinese embedding model (teacher) |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | ~80MB | Small embedding model (student base) |

### Fine-tuned Models (14 total)

| Type | Chunk | Margin | Count |
|------|-------|--------|-------|
| FFT | semantic | 0.3, 0.4, 0.5 | 3 |
| FFT | sliding | 0.3, 0.4, 0.5 | 3 |
| LoRA | semantic | 0.3, 0.4, 0.5 | 3 |
| LoRA | sliding | 0.3, 0.4, 0.5 | 3 |
| Base MiniLM | - | - | 1 |
| Base BGE-large | - | - | 1 |

### Reranker

- `BAAI/bge-reranker-v2-gemma`: Cross-Encoder reranking model (~8GB)

---

## Training Pipeline

### FFT (Full Parameter Fine-Tuning)

```bash
python scripts/train_minilm_FFT.py --chunk-method semantic --margin 0.3
python scripts/train_minilm_FFT.py --chunk-method sliding --margin 0.3
```

**Training config**:
- Batch size: 64
- Loss: TripletLoss
- Learning rate: Auto (LR Finder)
- Early stopping: patience=3

### LoRA Fine-Tuning

```bash
python scripts/train_minilm_LoRA.py --chunk-method semantic --margin 0.3
```

**LoRA config**:
- Rank: 16
- Alpha: 32
- Learning rate: 2e-4

### Margin Selection

Grid search for optimal margin (range [0.3, 0.4, 0.5]):

| Chunk | Best Margin | Validation Loss |
|-------|-------------|-----------------|
| semantic | 0.3 | 0.1731 |
| sliding | 0.3 | 0.1565 |

---

## FFT vs LoRA Comparison

| Metric | FFT | LoRA | Difference |
|--------|-----|------|------------|
| Best validation loss | 0.1565 | 0.1869 | FFT 16% lower |
| Training time | 340s | 213s | LoRA 39% faster |
| GPU memory | 6303MB | 4955MB | LoRA 22% less |
| Parameter update | All | Only adapter (r=16) |

---

## Evaluation System

### Quality Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Recall@k | hit_count / total_relevant | Proportion of relevant docs retrieved |
| HitRate@k | 1 if hit_count > 0 else 0 | Whether relevant doc found |
| MRR@k | 1 / first_rel_rank | Reciprocal rank of first relevant doc |
| nDCG@k | DCG / IDCG | Normalized measure considering ranking position |

### Efficiency Metrics

- Average retrieval latency (ms)
- Average rerank latency (ms)
- Average end-to-end latency (ms)
- Peak GPU memory (MB)

### Run Evaluation

```bash
# Single model evaluation
python src/evaluations/run_evaluation.py --model sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3 --chunk semantic

# Batch evaluation (all models)
python scripts/evaluation_results.py
```

---

## Web Interface

### Features

- **Model selection**: Dropdown to select embedding model
- **Chunk method**: Auto-display available methods based on model
- **Reranker toggle**: Button to enable/disable reranking
- **Real-time QA**: Input question to get RAG answer

### API Endpoints

| Endpoint | Method | Function |
|----------|--------|----------|
| `/api/models` | GET | Get available model list |
| `/api/config` | POST | Set model configuration |
| `/api/config` | GET | Get current configuration |
| `/api/rag` | POST | Handle query request |

### Response Format

```json
{
  "success": true,
  "answer": "Answer text",
  "sources": ["Source file 1", "Source file 2"],
  "efficiency": {
    "retrieval_ms": 21.5,
    "rerank_ms": 150.2,
    "total_ms": 171.7
  }
}
```

---

## Core Module Description

### Document Chunking (src/chunking/)

| Method | Description | Use Case |
|--------|-------------|----------|
| semantic | Separator-based semantic chunking | Maintain paragraph integrity |
| sliding_window | Fixed-size sliding window | Fine-grained coverage |

### Vector Store (src/embedding/vectorstore.py)

Custom model loading support:

```python
from embedding.vectorstore import (
    load_vectorstore_for_custom_model,
    search_custom_vectorstore,
    build_vectorstore_for_custom_model,
)
from embedding.embedder import get_custom_embedder

# Load fine-tuned model
vectorstore = load_vectorstore_for_custom_model(
    "sentence-transformers--all-MiniLM-L6-v2-FFT-semantic-0.3",
    "semantic"
)
```

### Reranking (src/reranking/)

```python
from reranking.reranker import load_reranker, rerank_chunks

reranker = load_reranker("BAAI/bge-reranker-v2-gemma")
reranked = rerank_chunks(query, chunks, top_k=5)
```

---

## Data Flow

```
Ingestion Pipeline:
Raw documents → Load → Chunk → Preprocess → Vectorize → Vector DB

Query Pipeline:
Question → Embed → Retrieve → [Rerank] → Build context → Generate answer
```

---

## Configuration Description

### configs.yaml

| Config | Description | Default |
|--------|-------------|---------|
| `chunking.chunk_size` | Chunk size | 300 |
| `chunking.chunk_overlap` | Chunk overlap | 100 |
| `retrieval.top_k` | Retrieval count | 5 |
| `models.generator_backend` | Generator backend | aliyun |

### configs_local.yaml (Sensitive)

| Config | Description |
|--------|-------------|
| `aliyun_dashscope_api_key` | Aliyun API key |
| `generation_api_key` | Query generation API key |

---

## Dependencies

Main dependencies (see requirements.txt):

- `sentence-transformers >= 2.2.0`
- `torch >= 2.0.0`
- `flask >= 2.0.0`
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`

---

## Notes

1. **Vector DB must exist**: Run `embed_to_vectordb.py` before selecting model
2. **LoRA models need merge**: Use `addition_ingestion.py` to merge LoRA weights before ingestion
3. **GPU memory**: Loading BGE-reranker requires ~8GB GPU memory
4. **API key security**: `configs_local.yaml` is not committed to Git

---

## Course Info

- Course: DDA4210 Tax Law Intelligent QA System
- Semester: 2026 Spring
- Project: MiniLM fine-tuned RAG system based on knowledge distillation

---

## License

MIT License