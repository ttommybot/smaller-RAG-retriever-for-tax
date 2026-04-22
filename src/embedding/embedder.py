# -*- coding: utf-8 -*-
"""
Text embedding module supporting large/small/student models with optional fine-tuned weights.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Install sentence-transformers: pip install sentence-transformers"
    )


def _load_config() -> Dict[str, Any]:
    """Load config from configs/configs.yaml."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    config_path = project_root / "configs" / "configs.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _get_local_model_path(model_name: str) -> Path:
    """Get local model path."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    models_dir = project_root / "models"
    return models_dir / model_name.replace("/", "--")


_large_model: Optional[SentenceTransformer] = None


def load_large_model() -> SentenceTransformer:
    """Load large embedding model."""
    global _large_model

    if _large_model is None:
        config = _load_config()
        model_name = config.get('embedding', {}).get('model_large_name', 'BAAI/bge-large-zh-v1.5')
        model_path = _get_local_model_path(model_name)

        print(f"Loading large model: {model_path}")
        _large_model = SentenceTransformer(str(model_path))
        print(f"Large model loaded")

    return _large_model


def embed_texts_large(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Encode texts with large model."""
    model = load_large_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_large(text: str, normalize: bool = True) -> np.ndarray:
    """Encode single query with large model."""
    embeddings = embed_texts_large([text], normalize=normalize)
    return embeddings[0]


_small_model: Optional[SentenceTransformer] = None


def load_small_model() -> SentenceTransformer:
    """Load small embedding model."""
    global _small_model

    if _small_model is None:
        config = _load_config()
        model_name = config.get('embedding', {}).get('model_small_name', 'sentence-transformers/all-MiniLM-L6-v2')
        model_path = _get_local_model_path(model_name)

        print(f"Loading small model: {model_path}")
        _small_model = SentenceTransformer(str(model_path))
        print(f"Small model loaded")

    return _small_model


def embed_texts_small(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Encode texts with small model."""
    model = load_small_model()
    texts = [str(t) for t in texts if t is not None]
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_small(text: str, normalize: bool = True) -> np.ndarray:
    """Encode single query with small model."""
    embeddings = embed_texts_small([str(text)], normalize=normalize)
    return embeddings[0]


_student_model: Optional[SentenceTransformer] = None


def load_student_model(use_finetuned: bool = False) -> SentenceTransformer:
    """Load student embedding model."""
    global _student_model

    if _student_model is None:
        config = _load_config()
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        if use_finetuned:
            model_path = project_root / "models" / "minilm-finetuned" / "final"
            print(f"Loading fine-tuned student model: {model_path}")
        else:
            model_name = config.get('embedding', {}).get('model_student_name', 'sentence-transformers/all-MiniLM-L6-v2')
            model_path = _get_local_model_path(model_name)
            print(f"Loading student model: {model_path}")

        _student_model = SentenceTransformer(str(model_path))
        print(f"Student model loaded")

    return _student_model


def embed_texts_student(texts: List[str], normalize: bool = True, use_finetuned: bool = False) -> np.ndarray:
    """Encode texts with student model."""
    model = load_student_model(use_finetuned=use_finetuned)
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def embed_query_student(text: str, normalize: bool = True, use_finetuned: bool = False) -> np.ndarray:
    """Encode single query with student model."""
    embeddings = embed_texts_student([text], normalize=normalize, use_finetuned=use_finetuned)
    return embeddings[0]


MODEL_TYPES = {
    'large': {
        'load': load_large_model,
        'embed_texts': embed_texts_large,
        'embed_query': embed_query_large
    },
    'small': {
        'load': load_small_model,
        'embed_texts': embed_texts_small,
        'embed_query': embed_query_small
    },
    'student': {
        'load': load_student_model,
        'embed_texts': embed_texts_student,
        'embed_query': embed_query_student
    }
}


def get_embedder(model_type: str = 'large') -> Dict[str, Any]:
    """Get embedder interface for specified model type."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}, available: {list(MODEL_TYPES.keys())}")

    return MODEL_TYPES[model_type]
