"""Shared paths and constants. All paths are relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ZIP = ROOT / "data" / "apple_dataset.zip"
RAW_DIR = ROOT / "data" / "raw" / "apple_disease_classification"
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASS_NAMES = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]
IMG_SIZE = (180, 180)
SEED = 42
