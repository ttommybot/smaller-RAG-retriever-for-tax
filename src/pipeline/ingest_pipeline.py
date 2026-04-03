import yaml
from pathlib import Path
from loading.loader import load_documents_from_dir
from loading.reprocess import clean_text 
from chunking.chunker import chunk_documents
from embedding.embedder import load_embedding_model, embed_texts
from embedding.vectorstore import build_vectorstore


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_ingestion_pipeline(config_path):
    """
    Build vector database from raw documents

    Args:
        config_path: Path to configuration file

    Returns:
        int: Number of documents processed
    """
    # Load configuration
    config = load_config(config_path)

    print(f"Loading config from: {config_path}")
    print(f"Raw data directory: {config['paths']['raw_data_dir']}")
    print(f"Processed data directory: {config['paths']['processed_data_dir']}")
    print(f"Vector DB directory: {config['paths']['vector_db_dir']}")

    # Load documents from directory
    print("\n[1/6] Loading documents from raw data directory...")
    raw_dir = Path(config['paths']['raw_data_dir'])
    documents = load_documents_from_dir(raw_dir)
    print(f"Loaded {len(documents)} documents")

    # Clean text
    print("\n[2/6] Cleaning text...")
    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc['text'])
        cleaned_docs.append({
            'text': cleaned_text,
            'metadata': doc.get('metadata', {})
        })
    print(f"Cleaned {len(cleaned_docs)} documents")

    # Chunk documents
    print("\n[3/6] Chunking documents...")
    chunk_size = config['chunking']['chunk_size']
    chunk_overlap = config['chunking']['chunk_overlap']
    chunks = chunk_documents(cleaned_docs, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    # Load embedding model
    print("\n[4/6] Loading embedding model...")
    embedding_model_name = config['models']['embedding_model_name']
    embedding_model = load_embedding_model(embedding_model_name)
    print(f"Loaded embedding model: {embedding_model_name}")

    # Embed texts
    print("\n[5/6] Generating embeddings for chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embed_texts(embedding_model, texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Build and save vector store
    print("\n[6/6] Building and saving vector store...")
    vector_db_dir = Path(config['paths']['vector_db_dir'])
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = build_vectorstore(embeddings, chunks, vector_db_dir)
    print(f"Vector store saved to: {vector_db_dir}")

    print(f"\n✓ Ingestion pipeline completed successfully!")
    print(f"  - Documents: {len(documents)}")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Embeddings: {embeddings.shape[0]}")

    return len(documents)
