import os

log_level = os.environ.get("LOG_LEVEL", "INFO")
cache_dir = os.environ.get("CACHE_DIR", "cache")

qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"), base=10)
