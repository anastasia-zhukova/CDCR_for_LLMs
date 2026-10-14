import os
from pathlib import Path

from dotenv import load_dotenv

from llm_ucdcr.logs import LogConfig, LogFormat, LogLevel

load_dotenv()

# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "uCDCR"
CONFIG_DIR = PROJECT_ROOT / "configs"

ENV = os.environ["ENV"]
LOG_FORMAT = os.environ["LOG_FORMAT"]
LOG_LEVEL = os.environ["LOG_LEVEL"]

LOG_CONFIG = LogConfig(format=LogFormat(LOG_FORMAT), level=LogLevel[LOG_LEVEL.upper()], env=ENV)

LLM_API_URL = os.environ["LLM_API_URL"]
LLM_API_TOKEN = os.environ["LLM_API_TOKEN"]
LLM_MODEL_NAME = os.environ["LLM_MODEL_NAME"]

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"]
MLFLOW_TRACKING_PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"]
