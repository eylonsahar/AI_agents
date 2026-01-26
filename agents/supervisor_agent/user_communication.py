from config import MANDATORY_INFO

class UserCommunication:
    def __init__(self):
        # Professional yet friendly prompts for the basics
        self.mandatory_info_prompts = {
            'max_price': '💰 What is your maximum budget for the vehicle?',
            'make': '🚘 What car make/brand are you looking for?',
            'model': '🏎️  What specific car model are you looking for?',
            'year_min': '🗓️  What is the minimum year you would consider?',
            'year_max': '🗓️  What is the maximum year you would consider?',
            'max_mileage': '🛣️  What is the maximum mileage you would accept?',
        }

    ###########################
    # Get info from user
    ###########################
    def get_vehicle_request(self) -> dict:
        """
        Collects mandatory requirements first, then natural language preferences.
        Returns a dictionary with keys: "required_info", "description", "full_query".
        """
        print("\n" + "═" * 40)
        print("   🤖 AItzik: Your AI Car Agent")
        print("   I'm here to help you find the best used car for your needs.")
        print("═" * 40)
        print("\nLet's start with the basics to narrow down the search:\n")

        vehicle_request = {}

        # 1. Collect Mandatory Info First
        for key in MANDATORY_INFO:
            prompt = self.mandatory_info_prompts.get(key, f"Please enter {key.replace('_', ' ')}:")
            while True:
                value = input(f"   {prompt} ").strip()
                if value:
                    vehicle_request[key] = value
                    break
                print(f"   ⚠️  This field is required to start the search.")

        # 2. Collect Natural Language Description Last
        print("\n" + "─" * 40)
        print("🎨 PERSONAL PREFERENCES")
        print("Now, tell me more about your lifestyle and preferences.")
        print("(e.g., color, fuel type, or specific features like 'leather seats')")
        description = input(" > ").strip()
        vehicle_request['description'] = description

        # 3. Concatenation: Create the flattened 'Super Query' for EmbeddingGateway
        query_parts = []
        for key in MANDATORY_INFO:
            query_parts.append(f"{key.replace('_', ' ')}: {vehicle_request[key]}")

        if description:
            query_parts.append(f"Additional Preferences: {description}")

        # Flattened query to be sent to EmbeddingGateway.embed_query()
        vehicle_request['full_query'] = " ".join(query_parts)

        print(f"\n✅ Request received! Searching the market for your {vehicle_request.get('make', '')}...")
        return vehicle_request


    ###########################
    # Return output to user
    ###########################
    def return_vehicle_ads(self, listings_groups: list[dict], meetings_list: list[list[dict]]) -> None:
        """
        Displays vehicle recommendations grouped by model to avoid redundancy.
        """
        print("\n" + "═" * 60)
        print("🌟           AItzik's TOP RECOMMENDATIONS             🌟")
        print("═" * 60 + "\n")

        if not listings_groups:
            print("I couldn't find any vehicles matching those exact criteria.")
            return

        ad_counter = 0

        # 1. Iterate through vehicle groups (e.g., Kia Sportage group)
        for group in listings_groups:
            vehicle_context = group.get("vehicle", {})
            listings = group.get("listings", [])

            # Print GROUP HEADER once per vehicle type
            print(f"\n{'━' * 60}")
            print(f" 🚘 MODEL: {vehicle_context.get('make', '').upper()} {vehicle_context.get('model', '').upper()}")
            print(f" 💡 WHY THIS MODEL: {vehicle_context.get('match_reason', 'Matches your criteria.')}")
            print(f"{'━' * 60}")

            # 2. Iterate through specific listings within this group
            for i, listing in enumerate(listings, 1):
                print(f"\n   📍 OPTION #{i}:")

                # Display specific listing details (Skipping redundant Make/Model)
                internal_keys = ['id', 'posting_date', 'state', 'manufacturer', 'model']
                for key, value in listing.items():
                    if key not in internal_keys:
                        display_key = key.replace('_', ' ').title()
                        print(f"      • {display_key}: {value}")

                # 3. Display specific meeting links for this ad
                if ad_counter < len(meetings_list):
                    current_links = meetings_list[ad_counter]
                    if current_links:
                        print(f"\n      📅 SCHEDULE VIEWING:")
                        for link_data in current_links:
                            # Extracts 'slot' and 'url' from FieldAgent's dict
                            print(f"         🔗 {link_data['slot']}: {link_data['url']}")
                    ad_counter += 1

        print("\n" + "═" * 60)
        print("  Click the links above to add these viewings to your calendar!")
        print("═" * 60 + "\n")
