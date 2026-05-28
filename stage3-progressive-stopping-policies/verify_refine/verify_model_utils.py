import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_HELMET_DIR = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "stage1-agenticlu-runtime",
    "AgenticLU-Modified",
    "HELMET",
)
if BASE_HELMET_DIR not in sys.path:
    sys.path.append(BASE_HELMET_DIR)

from model_utils import *  # noqa: F401,F403
