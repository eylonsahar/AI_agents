#!/usr/bin/env python3
"""
Debug script for Field Agent - Visualize the complete flow step by step.

This script creates sample data and runs the Field Agent with detailed logging
so you can see exactly how it works.
"""

import json
import os
from datetime import datetime
from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog


def create_sample_data():
    """Create sample vehicle listings with missing fields for testing."""
    
    sample_ads = {
        "results": [
            {
                "vehicle": {
                    "make": "Toyota",
                    "model": "Camry"
                },
                "listings": [
                    {
                        "id": "1001",
                        "price": "15000",
                        "year": "2018",
                        "manufacturer": "Toyota", 
                        "model": "Camry",
                        # Missing: mileage, accident, condition, paint_color, state
                    },
                    {
                        "id": "1002", 
                        "price": "18000",
                        "year": "2020",
                        "manufacturer": "Toyota",
                        "model": "Camry", 
                        "mileage": "35000",
                        # Missing: accident, condition, paint_color, state
                    }
                ]
            },
            {
                "vehicle": {
                    "make": "Honda",
                    "model": "Civic"
                },
                "listings": [
                    {
                        "id": "2001",
                        "price": "12000",
                        "year": "2017",
                        "manufacturer": "Honda",
                        "model": "Civic",
                        "mileage": "45000",
                        "accident": "none",
                        # Missing: condition, paint_color, state
                    }
                ]
            }
        ]
    }
    
    return sample_ads


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


def print_initial_state(ads_list):
    """Print the initial state of the listings."""
    print_section_header("INITIAL LISTINGS STATE")
    
    for result in ads_list.get("results", []):
        vehicle_info = result.get("vehicle", {})
        vehicle_name = f"{vehicle_info.get('make')} {vehicle_info.get('model')}"
        print(f"\n🚗 {vehicle_name}:")
        
        for listing in result.get("listings", []):
            listing_id = listing.get("id")
            print(f"  📋 Listing ID: {listing_id}")
            
            # Show all fields and highlight missing ones
            critical_fields = ["price", "year", "manufacturer", "model", "mileage", "accident", "condition", "paint_color", "state"]
            for field in critical_fields:
                value = listing.get(field, "❌ MISSING")
                if value == "❌ MISSING":
                    print(f"    {field}: {value}")
                else:
                    print(f"    {field}: {value}")


def main():
    """Main debug function to run Field Agent with visualization."""
    
    print_section_header("FIELD AGENT DEBUG SESSION")
    print("🚀 Starting Field Agent debug flow...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create sample data
    print_section_header("STEP 1: CREATING SAMPLE DATA")
    ads_list = create_sample_data()
    print_initial_state(ads_list)
    
    # Step 2: Initialize ActionLog for tracking
    print_section_header("STEP 2: INITIALIZING ACTION LOG")
    action_log = ActionLog()
    print("✅ ActionLog created - will track all LLM calls")
    
    # Step 3: Create Field Agent instance
    print_section_header("STEP 3: INITIALIZING FIELD AGENT")
    print("🤖 Creating FieldAgent with LangChain ReAct...")
    
    try:
        field_agent = FieldAgent(
            ads_list=ads_list,
            action_log=action_log,
            max_iterations=10  # Lower for debugging
        )
        print("✅ FieldAgent initialized successfully")
        print(f"📊 Agent will process {len(ads_list.get('results', []))} vehicle groups")
        
        # Show initial agent state
        print_section_header("STEP 4: AGENT INTERNAL STATE")
        initial_state = field_agent._get_current_state()
        print("📋 Current Agent State:")
        print(initial_state)
        
        # Step 5: Run the agent
        print_section_header("STEP 5: RUNNING FIELD AGENT")
        print("🔄 Starting autonomous processing...")
        print("👀 Watch the ReAct loop: Thought → Action → Observation")
        print("-" * 80)
        
        result = field_agent.process_listings()
        
        print_section_header("STEP 6: PROCESSING COMPLETE")
        print("✅ Field Agent finished processing!")
        print(f"📊 Final Stats: {json.dumps(result['stats'], indent=2)}")
        
        # Step 7: Show final listings state
        print_section_header("STEP 7: FINAL LISTINGS STATE")
        for result in result.get("results", []):
            vehicle_info = result.get("vehicle", {})
            vehicle_name = f"{vehicle_info.get('make')} {vehicle_info.get('model')}"
            print(f"\n🚗 {vehicle_name}:")
            
            for listing in result.get("listings", []):
                listing_id = listing.get("id")
                print(f"  📋 Listing ID: {listing_id}")
                
                # Show all fields
                all_fields = dict(listing)
                for key, value in all_fields.items():
                    if key == "meetings" and isinstance(value, list):
                        print(f"    {key}: {len(value)} meeting slots scheduled")
                        for i, meeting in enumerate(value[:2]):  # Show first 2 meetings
                            print(f"      Slot {i+1}: {meeting.get('slot', 'N/A')}")
                            print(f"      URL: {meeting.get('url', 'N/A')[:50]}...")
                    else:
                        print(f"    {key}: {value}")
        
        # Step 8: Show detailed action log
        print_section_header("STEP 8: DETAILED ACTION LOG")
        field_agent.print_action_log()
        
        # Step 9: Summary
        print_section_header("DEBUG SESSION SUMMARY")
        print("🎯 What you just witnessed:")
        print("  1. Field Agent identified missing fields in listings")
        print("  2. Used MockSeller to fill missing data via LLM calls")
        print("  3. Generated Google Calendar links for meetings")
        print("  4. Logged every LLM interaction for audit trail")
        print("  5. Used LangChain ReAct pattern for autonomous decision making")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 Debug session complete!")
        
    except Exception as e:
        print_section_header("ERROR OCCURRED")
        print(f"❌ Error: {str(e)}")
        print("🔍 Check the error details above for debugging")
        raise


if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in .env file or environment.")
        exit(1)
    
    main()
