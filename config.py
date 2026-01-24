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
