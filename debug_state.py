#!/usr/bin/env python3
"""
Debug script to check the actual state after agent runs.
"""

from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog
import json


def main():
    """Debug the actual state after agent execution."""
    
    # Sample data
    ads_list = {
        "results": [
            {
                "vehicle": {"make": "Toyota", "model": "Camry"},
                "listings": [
                    {
                        "id": "1001",
                        "price": "15000",
                        "year": "2018",
                        "manufacturer": "Toyota", 
                        "model": "Camry",
                    }
                ]
            }
        ]
    }
    
    print("🚀 Starting Field Agent debug...")
    action_log = ActionLog()
    
    # Create agent
    field_agent = FieldAgent(ads_list, action_log, max_iterations=3)
    
    print("📋 Before processing:")
    print(f"  Completed listings: {field_agent.completed_listings}")
    print(f"  Processed listings: {field_agent.processed_listings}")
    
    # Check initial listing state
    listing, _ = field_agent._find_listing_by_id("1001")
    print(f"  Initial listing data: {json.dumps(listing, indent=2)}")
    
    print("\n🔄 Running agent...")
    try:
        result = field_agent.process_listings()
        
        print("\n📊 After processing:")
        print(f"  Completed listings: {field_agent.completed_listings}")
        print(f"  Processed listings: {field_agent.processed_listings}")
        
        # Check final listing state
        listing, _ = field_agent._find_listing_by_id("1001")
        print(f"  Final listing data: {json.dumps(listing, indent=2)}")
        
        print(f"\n📈 Result stats: {json.dumps(result['stats'], indent=2)}")
        
        print("\n📋 Action log steps:")
        action_log.print_steps()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Even if there's an error, check the state
        print("\n📊 State after error:")
        print(f"  Completed listings: {field_agent.completed_listings}")
        print(f"  Processed listings: {field_agent.processed_listings}")
        
        listing, _ = field_agent._find_listing_by_id("1001")
        print(f"  Listing data: {json.dumps(listing, indent=2)}")


if __name__ == "__main__":
    main()
