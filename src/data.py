"""Load the committed split CSVs into arrays / tf.data pipelines."""
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, IMG_SIZE, ROOT, SPLITS_DIR


def load_split(split):
    """Return (images uint8 [N,H,W,3] RGB, integer labels [N], paths)."""
    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    images = np.stack([load_image(ROOT / p) for p in df.path])
    labels = df.label.map(CLASS_NAMES.index).to_numpy()
    return images, labels, df.path.tolist()


def load_image(path, size=IMG_SIZE):
    """Load an image as RGB uint8 and resize to `size`. Used by training AND the app."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size, Image.BILINEAR), dtype=np.uint8)
