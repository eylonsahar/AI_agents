import os
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
from config import EMBEDDING_DIMENSIONS

load_dotenv()

def initialize_pinecone(index_name: str, dimensions: int = EMBEDDING_DIMENSIONS):
    """
    Initialize Pinecone client and create/connect to index.
    
    Args:
        index_name: Name of the Pinecone index to create/connect to
        dimensions: Embedding dimensions (default from config)
    
    Returns:
        Pinecone index object
    """
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )



        print(f"Index '{index_name}' created successfully")
    else:
        print(f"Index '{index_name}' already exists")
    
    # Connect to index
    index = pc.Index(index_name)


    # Get index stats
    stats = index.describe_index_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total vectors: {stats['total_vector_count']}")
    print(f"  Dimension: {stats['dimension']}")
    
    return index


def delete_index(index_name: str):
    """
    Delete the Pinecone index (use with caution).
    
    Args:
        index_name: Name of the Pinecone index to delete
    """
    api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=api_key)
    
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully")
    else:
        print(f"Index '{index_name}' does not exist")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pinecone_setup.py <index_name>")
        sys.exit(1)
    
    index_name = sys.argv[1]
    print(f"Initializing Pinecone for index: {index_name}...")
    index = initialize_pinecone(index_name)
    print("\nPinecone setup complete!")
