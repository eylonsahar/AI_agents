#!/usr/bin/env python3
"""
Test the Field Agent running autonomously (using LangChain ReAct loop).
"""

from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog
import json


def main():
    """Run the Field Agent autonomously to process listings."""
    
    # Sample data with a listing
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
    
    print("🚀 Field Agent Autonomous Test")
    print("=" * 50)
    
    action_log = ActionLog()
    field_agent = FieldAgent(ads_list, action_log, max_iterations=10)
    
    print("📋 Initial listings state:")
    print(json.dumps(ads_list, indent=2))
    print("\n" + "=" * 50)
    print("🤖 Starting autonomous agent processing...")
    print("=" * 50 + "\n")
    
    # Run the agent autonomously
    result = field_agent.process_listings()
    
    print("\n" + "=" * 50)
    print("✅ Agent finished processing")
    print("=" * 50)
    
    print("\n📊 Results:")
    print(f"  Stats: {json.dumps(result['stats'], indent=2)}")
    
    print("\n📋 Final listings state:")
    print(json.dumps(result['results'], indent=2))
    
    print("\n📋 Action log:")
    action_log.print_steps()


if __name__ == "__main__":
    main()
