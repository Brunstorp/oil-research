"""Project configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

START_DATE = "2008-01-01"
END_DATE   = "2026-05-05"

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
EIA_API_KEY  = os.environ.get("EIA_API_KEY", "")

FRED_SERIES = {
    "S":   "DCOILWTICO",   # WTI Cushing spot
    "OVX": "OVXCLS",       # CBOE Crude Oil Volatility Index
}

EIA_FUTURES = {
    "F1": "RCLC1",
    "F2": "RCLC2",
    "F3": "RCLC3",
}

