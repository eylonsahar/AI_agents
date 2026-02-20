from gateways.llm_gateway import LLMGateway
from agents.prompts import MOCK_SELLER_SYSTEM_PROMPT, MOCK_SELLER_SCHEDULING_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class MockSeller:
    """
    Simulates a used car seller to provide missing advertisement data via LLM-generated responses.
    """

    def __init__(self, query_from_field_agent: str, info_system_prompt: str = MOCK_SELLER_SYSTEM_PROMPT,
                 sched_system_prompt: str = MOCK_SELLER_SCHEDULING_PROMPT):
        """
        Initialize the seller with the agent's inquiry and persona instructions.
        """
        self.llm = LLMGateway(api_key=OPENAI_API_KEY)
        self.query_from_field_agent = query_from_field_agent
        self.info_system_prompt = info_system_prompt
        self.sched_system_prompt = sched_system_prompt

    ##############################################
    # For tool 1: Fill in missing data
    ##############################################
    def get_missing_data(self):
        """
        Query the LLM to generate a response in a '<field>=<value>' format.

        Returns:
            tuple: (str: seller_response, dict: token_usage)
        """
        # Combine system prompt and agent query for the LLM call
        full_prompt = f"{self.info_system_prompt}\n\nAgent Query: {self.query_from_field_agent}"

        seller_response, usage = self.llm.call_llm(prompt=full_prompt)

        return seller_response, usage


    ##############################################
    # For tool 2: Return optional meeting slots
    ##############################################
    def get_available_dates(self):
        """
        Generates meeting slots specifically within the next 14 days.
        """

        # Use the specialized scheduling system prompt
        full_prompt = self.sched_system_prompt + self.query_from_field_agent

        seller_response, usage = self.llm.call_llm(prompt=full_prompt)

        slots = [s.strip() for s in seller_response.strip().split('\n') if s.strip()]
        return slots, usage
