from agents.field_agent.mock_seller import MockSeller
from datetime import datetime, timedelta
from config import NUM_AVAILABLE_DATES, MEETING_DURATION, MEETING_TIMEFRAME
import urllib.parse


class FieldAgent:
    """
    Coordinates the retrieval of missing vehicle data by communicating with a MockSeller.
    """

    def __init__(self, ads_list: dict):
        """
        Initialize with the vehicle search results list from the RAGRetriever.
        """
        self.ads_list = ads_list

    ##############################################
    # Tool 1: Get missing data from seller
    ##############################################
    def _parse_seller_response_to_dict(self, raw_seller_response: str) -> dict:
        """
        Converts the LLM's string output into a dictionary for safe access.
        """
        data_from_seller = {}
        for line in raw_seller_response.strip().split('\n'):
            if "=" in line:
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
        missing_keys = [k for k, v in ad_listing.items() if v is None or v == ""]
        fields_to_ask = list(set(["mileage", "accident"] + missing_keys))

        query = f"Provide these fields: {', '.join(fields_to_ask)} for listing ID {ad_listing.get('id')}"

        seller = MockSeller(query)
        raw_response, _ = seller.get_missing_data()

        return self._parse_seller_response_to_dict(raw_response), fields_to_ask

    def fill_missing_data(self, ad_listing: dict, keys_to_fill: list, seller_results: dict) -> dict:
        """
        Updates the listing using .get() to safely handle missing keys from the seller.
        """
        updated_ad = ad_listing.copy()
        for key in keys_to_fill:
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


    ##############################################
    # Tool 2: Schedule meetings in Google Calendar
    ##############################################
    def schedule_meeting(self, slot: str, ad_listing: dict):
        start_time = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        end_time = start_time + timedelta(minutes=MEETING_DURATION)

        # Format: YYYYMMDDTHHMMSS (No dashes or colons)
        format_str = "%Y%m%dT%H%M%S"
        dates_param = f"{start_time.strftime(format_str)}/{end_time.strftime(format_str)}"

        params = {
            "action": "TEMPLATE",
            "text": f"View Car: {ad_listing.get('manufacturer', '')} {ad_listing.get('model', '')}",
            "dates": dates_param,  # This ensures the specific slot is used
            "details": f"Car ID: {ad_listing.get('id')}",
            "sf": "true"
        }
        return f"https://www.google.com/calendar/render?{urllib.parse.urlencode(params)}"

    def create_calendar_events(self, ad_listing: dict, num_meeting_slots: int = NUM_AVAILABLE_DATES):
        """
        Creates a list of calendar event URLs in time slots received from the MockSeller.
        """
        today = datetime.now()
        two_weeks_later = today + timedelta(days=MEETING_TIMEFRAME)

        query = (
            f"Generate exactly {num_meeting_slots} available slots.\n"
            "CONSTRAINTS:\n"
            f"- Date Window: {today.strftime('%Y-%m-%d')} to {two_weeks_later.strftime('%Y-%m-%d')}.\n"
            "- Days: Sunday to Friday ONLY.\n"
            "- Hours: 10:00 to 19:00.\n"
            "- Minutes: :00, :15, :30, or :45.\n"
            "Return ONLY the list of slots, one per line."
        )

        seller = MockSeller(query_from_field_agent=query)
        available_slots, _ = seller.get_available_dates()

        slot_links = []
        for slot in available_slots:
            url = self.schedule_meeting(slot, ad_listing)
            slot_links.append({"slot": slot, "url": url})

        return slot_links
