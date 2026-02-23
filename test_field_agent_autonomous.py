#!/usr/bin/env python3
"""
Test the Field Agent running autonomously (using LangChain ReAct loop).
"""

from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog
import json


def main():
    """Run the Field Agent autonomously to process listings."""
    
    # Real search pipeline output - Mini listings for young couple
    ads_list = {
        "results": [
            {
                "vehicle": {"make": "Mini", "model": "Convertible"},
                "listings": [
                    {
                        "id": "7316164401",
                        "region": "boise",
                        "price": 18695,
                        "year": 2017,
                        "condition": "",
                        "paint_color": "white",
                        "state": "id",
                        "manufacturer": "mini",
                        "model": "convertible"
                    },
                    {
                        "id": "7316369262",
                        "region": "sf bay area",
                        "price": 25498,
                        "year": 2018,
                        "condition": "",
                        "paint_color": "green",
                        "state": "ca",
                        "manufacturer": "mini",
                        "model": "convertible"
                    },
                    {
                        "id": "7314729537",
                        "region": "orange county",
                        "price": 19989,
                        "year": 2017,
                        "condition": "",
                        "paint_color": "black",
                        "state": "ca",
                        "manufacturer": "mini",
                        "model": "convertible"
                    }
                ]
            },
            {
                "vehicle": {"make": "Mini", "model": "Roadster"},
                "listings": [
                    {
                        "id": "7313757163",
                        "region": "north jersey",
                        "price": 8600,
                        "year": 2012,
                        "condition": "",
                        "paint_color": "red",
                        "state": "nj",
                        "manufacturer": "mini",
                        "model": "roadster"
                    },
                    {
                        "id": "7313357806",
                        "region": "central nj",
                        "price": 8600,
                        "year": 2012,
                        "condition": "",
                        "paint_color": "red",
                        "state": "nj",
                        "manufacturer": "mini",
                        "model": "roadster"
                    },
                    {
                        "id": "7313253966",
                        "region": "south jersey",
                        "price": 8600,
                        "year": 2012,
                        "condition": "",
                        "paint_color": "red",
                        "state": "nj",
                        "manufacturer": "mini",
                        "model": "roadster"
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
