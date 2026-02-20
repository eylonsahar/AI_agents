from config import MANDATORY_INFO


class UserCommunication:
    def __init__(self):
        self.mandatory_info_prompts = {
            'max_price': '💰 What is your maximum budget for the vehicle?',
            'make': '🚘 What car make/brand are you looking for?',
            'model': '🎯 What specific car model are you looking for?',
            'year_min': '📅 What is the minimum year you would consider?',
            'year_max': '📅 What is the maximum year you would consider?',
            'max_mileage': '🛣️ What is the maximum mileage you would accept?',
        }

    ###########################
    # Get info from user
    ###########################
    def get_vehicle_request(self) -> dict:
        """
        Collects mandatory requirements first, then natural language preferences.
        Returns a dictionary with keys: "required_info", "description", "full_query".
        """
        print("\n" + "•" * 40)
        print("   🤖 AItzik: Your AI Car Agent")
        print("   I'm here to help you find the best used car for your needs.")
        print("•" * 40)
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
                print(f"   ⚠️ This field is required to start the search.")

        # 2. Collect Natural Language Description Last
        print("\n" + "━" * 40)
        print("🎨 PERSONAL PREFERENCES")
        print("Now, tell me more about your lifestyle and preferences.")
        print("(This could be about you or any other specific preferences like paint color)")
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
    # Display action log
    ###########################
    def _display_action_log(self, action_log: list[dict]) -> None:
        """
        Displays the detailed log of all agent actions for user transparency.
        """
        print("\n" + "=" * 80)
        print("📋 AGENT ACTION LOG - All Steps Performed")
        print("=" * 80)

        for i, step in enumerate(action_log, 1):
            print(f"\n[Step {i}] {step.get('module', 'Unknown')} → {step.get('submodule', 'Unknown')}")

            # Display prompt
            prompt = step.get('prompt', '')
            if len(prompt) > 150:
                print(f"        📤 Prompt: {prompt[:150]}...")
            else:
                print(f"        📤 Prompt: {prompt}")

            # Display response (truncate if too long)
            response = step.get('response', '')
            if len(response) > 150:
                print(f"        📥 Response: {response[:150]}...")
            else:
                print(f"        📥 Response: {response}")

            if step.get('timestamp'):
                print(f"        ⏱️  Time: {step['timestamp']}")

        print("\n" + "=" * 80)
        print(f"✅ Total LLM Calls: {len(action_log)}")
        print("=" * 80 + "\n")

    ###########################
    # Return output to user
    ###########################
    def return_vehicle_ads(self, listings_groups: list[dict], meetings_list: list[list[dict]],
                           action_log: list[dict] = None) -> None:
        """
        Displays vehicle recommendations grouped by model.
        Also displays the agent's detailed action log.

        Args:
            listings_groups: Vehicle listings grouped by model
            meetings_list: Calendar links for each listing
            action_log: Detailed log of all LLM calls made by the agent
        """

        print("\n" + "•" * 60)
        print("🌟           AItzik's TOP RECOMMENDATIONS             🌟")
        print("•" * 60 + "\n")

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
                print(f"\n   🏷️ OPTION #{i}:")

                # Display specific listing details (Skipping redundant Make/Model)
                listing_data = listing.get("listing_data", listing)
                internal_keys = ['id', 'posting_date', 'manufacturer', 'model']
                for key, value in listing_data.items():
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

        print("\n" + "•" * 60)
        print("  Click the links above to add these viewings to your calendar!")
        print("•" * 60 + "\n")

        # Display action log if provided
        if action_log:
            self._display_action_log(action_log)
