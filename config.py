# RAG System Hyperparameters Configuration


# Development settings
NUM_TEST_ROWS = 10  # Number of rows to use for testing/development

# Chunking parameters
CHUNK_SIZE = 800  # tokens (max 2048)
OVERLAP_RATIO = 0.2  # 20% overlap
TOP_K = 5  # number of chunks to retrieve (max 30)

# Agent configuration
MAX_RECOMMENDED_VEHICLES = 3  # Maximum number of vehicles to recommend


# OpenAI models
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_BASE_URL = "https://api.llmod.ai/v1"
EMBEDDING_DIMENSIONS = 1536
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
CHAT_BASE_URL = "https://api.llmod.ai/v1"
TOKENIZER_MODEL = "cl100k_base"

# Metadata columns to preserve (not embedded, only stored)
METADATA_COLUMNS = [
    'id',
    'make',
    'model',
    'body_type',
    'years',
    'url'

]
# Column to embed
EMBEDDED_COLUMN = 'all_text'


# Field agent configurations
NUM_AVAILABLE_DATES = 2 # Number of date+time options for a meeting with a seller
MEETING_DURATION = 30   # Duration of a meeting with a seller in minutes
MEETING_TIMEFRAME = 14  # Number of the following days to consider for a meeting with a seller
MANDATORY_INFO = ['max_price', 'year_min'] # Mandatory info to get from the user
GUARANTEED_MISSING_FIELDS = ["mileage", "accident"] # Fields that are guaranteed to be missing and must be filled
CRITICAL_FIELDS = [
    "price",
    "year",
    "condition",
    "mileage",
    "accident",
    "manufacturer",
    "model",
    "paint_color",
    "state"
]
MAX_DECISION_ITERATIONS = 50 # Max decision iteration to prevent infinite loop


# Supervisor configurations
NUM_TARGET_LISTINGS = 4