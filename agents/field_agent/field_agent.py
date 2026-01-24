from agents.field_agent.mock_seller import MockSeller


class FieldAgent:
    """
    Coordinates the retrieval of missing vehicle data by communicating with a MockSeller.
    """

    def __init__(self, ads_list: dict):
        """
        Initialize with the vehicle search results list from the RAGRetriever.
        """
        self.ads_list = ads_list

    def _parse_seller_response_to_dict(self, raw_seller_response: str) -> dict:
        """
        Converts the LLM's string output into a dictionary for safe access.
        """
        data_from_seller = {}
        for line in raw_seller_response.strip().split('\n'):
            if "=" in line:
                # Split only on the first '=' in case the value contains an '='
                # We will get exactly key-value pairs
                parts = line.split("=", 1)
                key = parts[0].strip().lower()
                value = parts[1].strip()
                data_from_seller[key] = value
        return data_from_seller

    def get_one_ad_data(self, ad_listing: dict):
        """
        Determines missing fields and contacts the MockSeller to provide the requested data.
        Returns a dictionary of the seller's response and a list of requested keys.
        """
        # Identify missing or empty fields
        missing_keys = [k for k, v in ad_listing.items() if v is None or v == ""]

        fields_to_ask = list(set(["mileage", "accident"] + missing_keys))

        # TODO: make this query use a natural language template
        query = f"Provide these fields: {', '.join(fields_to_ask)} for listing ID {ad_listing.get('id')}"

        seller = MockSeller(query)
        raw_response, _ = seller.get_seller_response()

        return self._parse_seller_response_to_dict(raw_response), fields_to_ask

    def fill_missing_data(self, ad_listing: dict, keys_to_fill: list, seller_results: dict) -> dict:
        """
        Updates the listing using .get() to safely handle missing keys from the seller.
        """
        updated_ad = ad_listing.copy()
        for key in keys_to_fill:
            # Use .get() to provide a fallback value if the LLM missed a field
            updated_ad[key] = seller_results.get(key, "Not Provided")
        return updated_ad

    def get_all_ads_data(self) -> list:
        """
        Processes the entire payload and returns a list of completed ad data.
        """
        final_results = []
        for result in self.ads_list.get("results", []):
            vehicle_context = result.get("vehicle")
            updated_listings = []

            for listing in result.get("listings", []):
                seller_dict, keys_requested = self.get_one_ad_data(listing)
                complete_listing = self.fill_missing_data(listing, keys_requested, seller_dict)
                updated_listings.append(complete_listing)

            final_results.append({
                "vehicle": vehicle_context,
                "listings": updated_listings
            })
        return final_results
