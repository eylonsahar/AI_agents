"""
Manual Test Script for VehicleModelRetriever

This script allows you to test and debug the VehicleModelRetriever
with different queries and configurations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from dotenv import load_dotenv
from pinecone import Pinecone
from gateways import EmbeddingGateway, LLMGateway
from agents.search_agents.vehicle_model_retriever import VehicleModelRetriever
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    CHAT_MODEL,
    CHAT_BASE_URL
)
import json

# Load environment variables
load_dotenv()


def initialize_gateways():
    """Initialize embedding and LLM gateways."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize Embedding Gateway
    embedding_gateway = EmbeddingGateway(
        api_key=openai_api_key,
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL
    )
    
    # Initialize LLM Gateway
    llm_gateway = LLMGateway(
        api_key=openai_api_key,
        model=CHAT_MODEL,
        base_url=CHAT_BASE_URL
    )
    
    return embedding_gateway, llm_gateway


def initialize_pinecone():
    """Initialize Pinecone and get the index."""
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Get the index (adjust index name as needed)
    index_name = 'vehicles-info'  # Change this to your actual index name
    index = pc.Index(index_name)
    
    print(f"Connected to Pinecone index: {index_name}")
    print(f"Index stats: {index.describe_index_stats()}")
    
    return index


def test_vehicle_retriever(query: str, max_retries: int = 3):
    """
    Test the VehicleModelRetriever with a specific query.
    
    Args:
        query: User query to test
        top_n: Number of vehicles to retrieve
        max_retries: Maximum retry attempts for validation
    """
    print("\n" + "="*80)
    print(f"TESTING VEHICLE MODEL RETRIEVER")
    print("="*80)
    print(f"\nQuery: {query}")
    print(f"Max Retries: {max_retries}")
    print("\n" + "-"*80)
    
    try:
        # Initialize components
        print("\n[1/4] Initializing gateways...")
        embedding_gateway, llm_gateway = initialize_gateways()
        print("✓ Gateways initialized")
        
        print("\n[2/4] Connecting to Pinecone...")
        pinecone_index = initialize_pinecone()
        print("✓ Pinecone connected")
        
        print("\n[3/4] Creating VehicleModelRetriever...")
        retriever = VehicleModelRetriever(
            pinecone_index=pinecone_index,
            embedding_gateway=embedding_gateway,
            llm_gateway=llm_gateway,
            max_retries=max_retries
        )
        print("✓ Retriever created")
        
        print("\n[4/4] Searching for vehicle models...")
        print("-"*80)
        
        result = retriever.search_vehicle_models(query=query)
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        # Pretty print the result
        print(f"\n📊 Number of vehicles returned: {len(result.get('vehicles', []))}")
        print(f"\n💬 Explanation:\n{result.get('explanation', 'No explanation provided')}")
        
        if result.get('vehicles'):
            print(f"\n🚗 Recommended Vehicles:\n")
            for i, vehicle in enumerate(result['vehicles'], 1):
                print(f"\n{i}. {vehicle.get('make', 'N/A')} {vehicle.get('model', 'N/A')}")
                print(f"   Body Type: {vehicle.get('body_type', 'N/A')}")
                print(f"   Years: {vehicle.get('years', 'N/A')}")
                print(f"   Match Score: {vehicle.get('match_score', 0):.2f}")
                print(f"   Reason: {vehicle.get('match_reason', 'N/A')}")
        else:
            print("\n⚠️  No vehicles found or validation failed")
        
        # Save result to file for inspection
        output_file = 'test_result.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full result saved to: {output_file}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    """Run in interactive mode for multiple queries."""
    print("\n" + "="*80)
    print("VEHICLE MODEL RETRIEVER - INTERACTIVE TEST MODE")
    print("="*80)
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'help' for example queries\n")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n📝 Example queries:")
                print("  - I need a reliable SUV for family trips")
                print("  - Looking for a fuel-efficient sedan under $30k")
                print("  - Best luxury cars for long commutes")
                print("  - Affordable compact car for city driving")
                print("  - Spacious minivan for large family")
                continue
            
            if not query:
                print("⚠️  Please enter a query")
                continue
            
            # Get optional parameters
            try:
                top_n = input("📊 Number of vehicles to retrieve (default 10): ").strip()
                top_n = int(top_n) if top_n else 10
            except ValueError:
                top_n = 10
            
            # Run test
            test_vehicle_retriever(query=query, top_n=top_n)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test VehicleModelRetriever')
    parser.add_argument('--query', '-q', type=str, help='Query to test')
    parser.add_argument('--top-n', '-n', type=int, default=10, help='Number of vehicles to retrieve')
    parser.add_argument('--max-retries', '-r', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        test_vehicle_retriever(
            query=args.query,
            top_n=args.top_n,
            max_retries=args.max_retries
        )
    else:
        # Default test queries
        test_queries = [
            "I need a reliable SUV for family trips",
            "Looking for a fuel-efficient sedan",
            "Best luxury cars for long commutes"
        ]
        
        print("\n🧪 Running default test queries...\n")
        for query in test_queries:
            test_vehicle_retriever(query=query, top_n=5)
            print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Direct test call for debugging - uncomment and modify as needed
    test_vehicle_retriever(
        query="I’m looking for a reliable compact car for a young couple with a low budget.",
        max_retries=3
    )

