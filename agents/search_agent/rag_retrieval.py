from typing import List, Dict, Tuple, Callable, Optional
from gateways import LLMGateway, EmbeddingGateway
from config import NUM_OF_CHUNKS_TO_RETRIEVE, PINECONE_INDEX_NAME, EMBEDDING_DIMENSIONS
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


class RAGRetriever:
    """
    Generic RAG retrieval system.
    Handles query embedding, similarity search, and response generation.
    
    This class is now domain-agnostic and can be used for any RAG application
    by providing custom system prompts and context formatters.
    """
    
    def __init__(
        self,
        pinecone_index,
        embedding_gateway: EmbeddingGateway,
        llm_gateway: Optional[LLMGateway] = None,
        system_prompt: Optional[str] = None,
        context_formatter: Optional[Callable[[List[Dict]], str]] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            pinecone_index: Pinecone index object
            embedding_gateway: EmbeddingGateway instance (required)
            llm_gateway: Optional LLM gateway for response generation
            system_prompt: Optional custom system prompt
            context_formatter: Optional function to format retrieved chunks into context
        """
        self.index = pinecone_index
        self.embedding_gateway = embedding_gateway
        self.llm_gateway = llm_gateway
        self.system_prompt = system_prompt
        self.context_formatter = context_formatter
    
    def search_similar_chunks(self, query: str, top_k: int = NUM_OF_CHUNKS_TO_RETRIEVE) -> List[Dict]:
        """
        Search for similar chunks in Pinecone.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of matched chunks with metadata and scores
        """
        query_embedding = self.embedding_gateway.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        chunks = []
        for match in results['matches']:
            chunks.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })
        
        return chunks

    def retrieve_context(self, user_query: str, top_k: int = NUM_OF_CHUNKS_TO_RETRIEVE) -> Tuple[List[Dict], str]:
        chunks = self.search_similar_chunks(user_query, top_k=top_k)

        if not chunks:
            return [], ""

        context = self.context_formatter(chunks)
        return chunks, context
    
    def generate_response(self, query: str, context: str) -> Tuple[str, Dict]:
        """
        Generate response using LLM gateway with retrieved context.
        
        Args:
            query: User question
            context: Formatted context from retrieved chunks
        
        Returns:
            Tuple of (response text, usage stats)
        """
        if self.llm_gateway is None:
            return "Response generation disabled", {}
        

        full_prompt = f"{self.system_prompt}\n\nContext:\n\n{context}\n\nQuestion: {query}"

        
        return self.llm_gateway.call_llm(prompt=full_prompt)
    
    def query(self, user_query: str, top_k: int = NUM_OF_CHUNKS_TO_RETRIEVE) -> Dict:
        """
        Main RAG query function.
        Retrieves relevant chunks and generates response.
        
        Args:
            user_query: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with response, chunks, prompt, and usage stats
        """
        chunks = self.search_similar_chunks(user_query, top_k=top_k)
        
        if not chunks:
            return {'response': "No results found", 'chunks': [], 'prompt': '', 'usage': {}}
        
        context = self.context_formatter(chunks)
        full_prompt = f"{self.system_prompt}\n\nContext:\n\n{context}\n\nQuestion: {user_query}"
        response, usage = self.generate_response(user_query, context)
        
        return {
            'response': response,
            'chunks': chunks,
            'prompt': full_prompt,
            'usage': usage
        }


def get_pinecone_index(index_name: str = None, dimensions: int = EMBEDDING_DIMENSIONS):
    """
    Initialize Pinecone and get the index object.
    
    Args:
        index_name: Name of the Pinecone index (default from config)
        dimensions: Embedding dimensions (default from config)
    
    Returns:
        Pinecone index object
    """
    if index_name is None:
        index_name = PINECONE_INDEX_NAME
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        raise ValueError(f"Index '{index_name}' not found. Available indexes: {existing_indexes}")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Verify index stats
    stats = index.describe_index_stats()
    print(f"Connected to Pinecone index '{index_name}'")
    print(f"Index stats: {stats}")
    
    return index

