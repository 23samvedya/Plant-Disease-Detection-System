"""Build a reproducible, leakage-free 70/15/15 stratified split.

The original 2023 dataset shipped as Train/ and Test/ folders. That split had
two problems:
  * exact duplicate images (same bytes, different file names), some of which
    appeared in BOTH Train and Test, and
  * no separate validation set, so the test set was used for model selection.

This script pools every image, removes exact duplicates (keeping one copy),
groups near-duplicates with a perceptual hash so they always land in the same
split, and then does a stratified, group-aware 70/15/15 split with a fixed seed.
The resulting file lists are written to data/splits/*.csv and committed, so the
split can be reproduced without re-running this script.

Usage:  python -m src.make_split
"""
import hashlib
import random
import zipfile
from collections import defaultdict

import imagehash
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, DATA_ZIP, RAW_DIR, SEED, SPLITS_DIR

PHASH_MAX_DISTANCE = 4  # Hamming distance at which two images count as near-duplicates
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def ensure_extracted():
    if not RAW_DIR.exists():
        with zipfile.ZipFile(DATA_ZIP) as zf:
            zf.extractall(RAW_DIR.parent)


def collect_images():
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = path.parent.name
        if label not in CLASS_NAMES:
            continue
        data = path.read_bytes()
        with Image.open(path) as im:
            phash = imagehash.phash(im.convert("RGB"))
        rows.append({
            "path": path.relative_to(RAW_DIR.parent.parent.parent).as_posix(),
            "label": label,
            "original_folder": path.parent.parent.name,
            "md5": hashlib.md5(data).hexdigest(),
            "phash": phash,
        })
    return rows


def group_near_duplicates(rows):
    """Union-find over perceptual hashes, so near-duplicates share a group id."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["phash"] - rows[j]["phash"] <= PHASH_MAX_DISTANCE:
                parent[find(i)] = find(j)
    for i, r in enumerate(rows):
        r["group"] = find(i)


def main():
    ensure_extracted()
    rows = collect_images()
    n_total = len(rows)

    # 1. Drop exact duplicates (keep the first copy in sorted order).
    seen, unique = {}, []
    for r in rows:
        if r["md5"] in seen:
            if seen[r["md5"]] != r["label"]:
                raise ValueError(f"Same image found under two labels: {r['path']}")
            continue
        seen[r["md5"]] = r["label"]
        unique.append(r)

    # 2. Group near-duplicates.
    group_near_duplicates(unique)
    groups = defaultdict(list)
    for r in unique:
        groups[r["group"]].append(r)
    # A near-duplicate group with conflicting labels is the same photo labelled
    # two different ways; we cannot tell which is right, so exclude it.
    excluded = []
    for gid, g in list(groups.items()):
        if len({r["label"] for r in g}) > 1:
            excluded += [{"path": r["path"], "label": r["label"],
                          "reason": "near-duplicate of an image with a different label"} for r in g]
            del groups[gid]
    unique = sorted((r for g in groups.values() for r in g), key=lambda r: r["path"])
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(excluded, columns=["path", "label", "reason"]).to_csv(
        SPLITS_DIR / "excluded.csv", index=False)

    # 3. Stratified, group-aware split: for each class, shuffle its groups and
    #    fill train/val/test up to the target ratios.
    rng = random.Random(SEED)
    for label in CLASS_NAMES:
        label_groups = [g for g in groups.values() if g[0]["label"] == label]
        rng.shuffle(label_groups)
        n = sum(len(g) for g in label_groups)
        targets = {"test": round(n * RATIOS["test"]), "val": round(n * RATIOS["val"])}
        counts = {"test": 0, "val": 0}
        for g in label_groups:
            split = "train"
            for s in ("test", "val"):
                if counts[s] + len(g) <= targets[s]:
                    split = s
                    counts[s] += len(g)
                    break
            for r in g:
                r["split"] = split

    df = pd.DataFrame(unique).drop(columns=["phash", "group"])
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in RATIOS:
        df[df.split == split][["path", "label", "original_folder", "md5"]].to_csv(
            SPLITS_DIR / f"{split}.csv", index=False)

    summary = df.pivot_table(index="label", columns="split", values="path",
                             aggfunc="count", fill_value=0)[["train", "val", "test"]]
    summary.loc["Total"] = summary.sum()
    summary["all"] = summary.sum(axis=1)
    summary.to_csv(SPLITS_DIR / "summary.csv")
    n_groups = len(groups)
    print(f"Images found: {n_total}; excluded for label conflicts: {len(excluded)}; final: {len(unique)}"
          f"; near-duplicate groups: {n_groups}")
    print(summary)


if __name__ == "__main__":
    main()
