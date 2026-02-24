#!/usr/bin/env python3
"""
Working debug script that bypasses ReAct parsing issues.
"""

from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog
import json


def main():
    """Debug the Field Agent by calling tools directly."""
    
    # Sample data
    ads_list = {
        "results": [
            {
                "vehicl "
                "e": {"make": "Toyota", "model": "Camry"},
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
    
    print("🚀 Field Agent Direct Tool Debug")
    print("=" * 50)
    
    action_log = ActionLog()
    field_agent = FieldAgent(ads_list, action_log, max_iterations=5)
    
    print("📋 Initial state:")
    listing, _ = field_agent._find_listing_by_id("1001")
    print(f"  Listing: {json.dumps(listing, indent=2)}")
    
    print("\n🔧 Step 1: Fill missing data")
    result1 = field_agent._tool_fill_missing_data(
        "1001", 
        ["mileage", "accident", "condition", "paint_color", "state"]
    )
    print(f"  Result: {result1}")
    
    print("\n📋 After filling data:")
    listing, _ = field_agent._find_listing_by_id("1001")
    print(f"  Listing: {json.dumps(listing, indent=2)}")
    
    print("\n🔧 Step 2: Schedule meeting")
    result2 = field_agent._tool_schedule_meeting("1001")
    print(f"  Result: {result2}")
    
    print("\n📋 After scheduling:")
    listing, _ = field_agent._find_listing_by_id("1001")
    print(f"  Listing: {json.dumps(listing, indent=2)}")
    
    print("\n🔧 Step 3: Complete processing")
    result3 = field_agent._tool_complete_processing()
    print(f"  Result: {result3}")
    
    print("\n📊 Final state:")
    print(f"  Completed listings: {field_agent.completed_listings}")
    print(f"  Processed listings: {field_agent.processed_listings}")
    
    print("\n📋 Action log:")
    action_log.print_steps()
    
    print("\n✅ Manual workflow complete!")
    print("This shows how the Field Agent tools work when called directly.")


if __name__ == "__main__":
    main()
