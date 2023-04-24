import json
import os
from typing import Optional

log_level = os.environ.get("LOG_LEVEL", "INFO")
cache_dir = os.environ.get("CACHE_DIR", "cache")
data_dir = os.environ.get("DATA_DIR", "data")

qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"), base=10)

default_limit = 5


project_config_file = os.path.join(data_dir, "projects.json")


# TODO: refactor
def set_remote(project: str, url: str):
    if os.path.exists(project_config_file):
        with open(project_config_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if project not in data:
        data[project] = {}

    data[project]["remote"] = url

    with open(project_config_file, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# TODO: refactor
def get_remote(project: str) -> Optional[str]:
    if os.path.exists(project_config_file):
        with open(project_config_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if project in data:
                return data[project]["remote"]

    return None
