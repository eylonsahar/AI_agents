from typing import List, Dict, Tuple, Callable, Optional
from gateways import LLMGateway, EmbeddingGateway
from config import TOP_K


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
    
    def search_similar_chunks(self, query: str, top_k: int = TOP_K) -> List[Dict]:
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
    
    def query(self, user_query: str, top_k: int = TOP_K) -> Dict:
        """
        Main RAG query function.
        Retrieves relevant chunks and generates response.
        
        Args:
            user_query: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with response, chunks, and usage stats
        """
        chunks = self.search_similar_chunks(user_query, top_k=top_k)
        
        if not chunks:
            return {'response': "No results found", 'chunks': [], 'usage': {}}
        
        context = self.context_formatter(chunks)
        response, usage = self.generate_response(user_query, context)
        
        return {
            'response': response,
            'chunks': chunks,
            'usage': usage
        }

