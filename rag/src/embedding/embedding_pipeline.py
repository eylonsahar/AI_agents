from dotenv import load_dotenv
from rag.src.embedding.pinecone_setup import initialize_pinecone
from gateways import EmbeddingGateway
import time
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS,
    NUM_TEST_ROWS,
    CHUNK_SIZE,
    OVERLAP_RATIO,
    METADATA_COLUMNS,
    EMBEDDED_COLUMN,
    TOKENIZER_MODEL,
)
import os
from tqdm import tqdm
import json
import tiktoken
from typing import List, Dict


load_dotenv()

def create_embeddings(texts: List[str], embedding_gateway: EmbeddingGateway) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using EmbeddingGateway.
    
    Args:
        texts: List of text strings to embed
        embedding_gateway: EmbeddingGateway instance
    
    Returns:
        List of embedding vectors
    """
    embeddings = embedding_gateway.embed_documents(texts)
    return embeddings


def upload_to_pinecone(chunks: List[Dict], index, embedding_gateway: EmbeddingGateway, batch_size: int = 50):
    """
    Embed chunks and upload to Pinecone in batches.
    
    Args:
        chunks: List of chunk dictionaries
        index: Pinecone index object
        embedding_gateway: EmbeddingGateway instance
        batch_size: Number of chunks to process at once
    """
    print(f"\nEmbedding and uploading {len(chunks)} chunks to Pinecone...")
    print(f"Batch size: {batch_size}")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i + batch_size]
        
        # Extract texts and prepare metadata
        texts = [chunk['text'] for chunk in batch]
        ids = [chunk['id'] for chunk in batch]
        metadatas = [chunk['metadata'] for chunk in batch]
        
        # Create embeddings
        embeddings = create_embeddings(texts, embedding_gateway)
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk_id, embedding, metadata, text in zip(ids, embeddings, metadatas, texts):
            # Add the chunk text to metadata for retrieval
            metadata_with_text = {**metadata, 'all_text': text}
            vectors.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': metadata_with_text
            })
        
        # Upload to Pinecone
        index.upsert(vectors=vectors)
        
        # Small delay to avoid rate limits
        time.sleep(0.1)
    
    print(f"\n✅ Successfully uploaded {len(chunks)} chunks to Pinecone")


def create_chunks_from_text(transcript: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Split transcript into overlapping chunks based on token count.

    Args:
        transcript: Full transcript text
        chunk_size: Maximum tokens per chunk
        overlap_ratio: Ratio of overlap between chunks (0-0.3)

    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding(TOKENIZER_MODEL)
    tokens = encoding.encode(transcript)

    chunks = []
    overlap_size = int(chunk_size * overlap_ratio)
    stride = chunk_size - overlap_size

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break
        start += stride

    return chunks


def load_vehicle_data(file_path: str, num_rows: int = None) -> List[Dict]:
    """Load vehicle data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_rows is not None and i >= num_rows:
                break
            data.append(json.loads(line))
    return data

def prepare_vehicle_chunks(vehicles: List[Dict]) -> List[Dict]:
    """Process vehicle data into chunks with metadata."""
    chunks = []
    for vehicle in tqdm(vehicles, desc="Processing vehicles"):
        # Get the text to be chunked
        text = str(vehicle.get(EMBEDDED_COLUMN, ''))
        if not text.strip():
            continue
            

        text_chunks = create_chunks_from_text(text, CHUNK_SIZE, OVERLAP_RATIO)
        
        # Create metadata from the vehicle data
        metadata = {}
        for col in METADATA_COLUMNS:
            if col in vehicle:
                metadata[col] = str(vehicle[col])
        
        # Create chunks with metadata
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_id': f"{vehicle.get('id', '')}_{i}"
            })
            
            chunks.append({
                'id': chunk_metadata['chunk_id'],
                'text': chunk_text,
                'metadata': chunk_metadata
            })
    
    return chunks


def _get_embedding_gateway() -> EmbeddingGateway:
    """Factory to return the configured embedding gateway.
    
    Returns:
        EmbeddingGateway instance configured with settings from config
    """
    return EmbeddingGateway(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL
    )


def main(
    index_name: str,
    data_path: str,
    use_full_dataset: bool = False,
    dimensions: int = EMBEDDING_DIMENSIONS
):
    """
    Main pipeline: Load vehicle data, chunk, embed, and upload to Pinecone.
    
    Args:
        index_name: Name of the Pinecone index to use
        data_path: Path to the data file (JSONL format)
        use_full_dataset: If True, process entire dataset. If False, use NUM_TEST_ROWS.
        dimensions: Embedding dimensions (default from config)
    """
    # Initialize LangChain embeddings client
    print("="*80)
    print("EMBEDDING PIPELINE")
    print("="*80)
    print(f"  Index: {index_name}")
    print(f"  Data: {data_path}")
    print(f"  Dimensions: {dimensions}")
    print("\n[1/5] Initializing embeddings client...")
    print("  Provider: EmbeddingGateway (OpenAI)")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Base URL: {EMBEDDING_BASE_URL}")

    embedding_gateway = _get_embedding_gateway()
    print("  ✅ Embedding gateway initialized")
    # Load data
    print(f"\n[2/5] Loading data...")
    print(f"  Data path: {data_path}")
    
    if use_full_dataset:
        print("  Loading FULL dataset...")
        data = load_vehicle_data(data_path)
    else:
        print(f"  Loading first {NUM_TEST_ROWS} rows for testing...")
        data = load_vehicle_data(data_path, num_rows=NUM_TEST_ROWS)
    
    print(f" ✅ Loaded {len(data)} records")

    # Prepare chunks
    print(f"\n[3/5] Preparing chunks...")
    chunks = prepare_vehicle_chunks(data)
    print(f"  ✅ Created {len(chunks)} chunks")
    
    # Show sample chunks
    if chunks:
        print("\nSample chunk:")
        print(f"  ID: {chunks[0]['id']}")
        print(f"  Text length: {len(chunks[0]['text'])} chars")
        print(f"  Metadata keys: {list(chunks[0]['metadata'].keys())}")
    
    # Initialize Pinecone
    print(f"\n[4/5] Initializing Pinecone...")
    index = initialize_pinecone(index_name, dimensions)
    print("  ✅ Pinecone ready")
    
    # Upload to Pinecone
    print(f"\n[5/5] Uploading to Pinecone...")
    upload_to_pinecone(chunks, index, embedding_gateway, batch_size=50)
    
    # Verify upload
    stats = index.describe_index_stats()
    print(f"\n✅ Verification: Index now contains {stats['total_vector_count']} vectors")


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python embedding_pipeline.py <index_name> <data_path> [--full]")
    #     print("Example: python embedding_pipeline.py vehicle-info-rag data/all_cars_data.jsonl")
    #     sys.exit(1)
    #
    # index_name = sys.argv[1]
    # data_path = sys.argv[2]
    # use_full = '--full' in sys.argv
    #
    main(
        index_name='vehicles-info',
        data_path="/Users/esahar/Documents/private/project/AI_agents/rag/data/all_cars_data.jsonl",
        use_full_dataset=True
    )
