#!/usr/bin/env python3
"""
Simple debug script - Quick way to see Field Agent in action.
"""

from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog


def main():
    """Quick debug run with minimal output."""
    
    # Sample data with missing fields
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
                        # Missing: mileage, accident, condition, paint_color, state
                    }
                ]
            }
        ]
    }
    
    print("🚀 Starting Field Agent debug...")
    action_log = ActionLog()
    
    # Create and run agent
    field_agent = FieldAgent(ads_list, action_log, max_iterations=5)
    
    print("📋 Initial state:")
    print(field_agent._get_current_state())
    
    print("\n🔄 Running agent...")
    result = field_agent.process_listings()
    
    print("\n✅ Results:")
    print(f"Completed listings: {result['stats']['completed_listings']}")
    
    print("\n📊 Action log:")
    field_agent.print_action_log()


if __name__ == "__main__":
    main()
