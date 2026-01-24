from gateways.llm_gateway import LLMGateway
from agents.prompts import MOCK_SELLER_SYSTEM_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class MockSeller:
    """
    Simulates a used car seller to provide missing advertisement data via LLM-generated responses.
    """

    def __init__(self, query_from_field_agent: str, system_prompt: str = MOCK_SELLER_SYSTEM_PROMPT):
        """
        Initialize the seller with the agent's inquiry and persona instructions.
        """
        self.llm = LLMGateway(api_key=OPENAI_API_KEY)
        self.query_from_field_agent = query_from_field_agent
        self.system_prompt = system_prompt

    def get_seller_response(self):
        """
        Query the LLM to generate a response in a '<field>=<value>' format.

        Returns:
            tuple: (str: seller_response, dict: token_usage)
        """
        # Combine system prompt and agent query for the LLM call
        full_prompt = f"{self.system_prompt}\n\nAgent Query: {self.query_from_field_agent}"

        seller_response, usage = self.llm.call_llm(prompt=full_prompt)
        return seller_response, usage
