"""Companion loader, reconstructed from the supplied inline pairing logic."""
import csv
import pickle
from pathlib import Path

import numpy as np


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"slide_id", "label", "high_pkl", "low_pkl"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        if row["slide_id"] in seen:
            raise ValueError(f"Duplicate slide_id: {row['slide_id']}")
        seen.add(row["slide_id"])
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1):
            raise ValueError("Labels must be 0 or 1")
        for key in ("high_pkl", "low_pkl"):
            source = Path(row[key])
            row[key] = source if source.is_absolute() else path.parent / source
            if not row[key].is_file():
                raise FileNotFoundError(row[key])
    return rows


def load_pair(high_path, low_path, dim=768):
    # Load only trusted experiment pickle files.
    with open(high_path, "rb") as f:
        high = pickle.load(f)
    with open(low_path, "rb") as f:
        low_rows = pickle.load(f)
    if not high:
        raise ValueError("High bag must contain at least one instance")
    low = {item["coord"]: item["feature"] for item in low_rows}
    if len(low) != len(low_rows):
        raise ValueError("Duplicate Low coordinates: resolve before pairing")
    features = np.zeros((len(high), 2 * dim), dtype=np.float32)
    coords = np.zeros((len(high), 5), dtype=np.float32)
    for i, item in enumerate(high):
        h = np.asarray(item["feature"], dtype=np.float32)
        if h.shape != (dim,) or not np.isfinite(h).all():
            raise ValueError(f"Invalid High feature at index {i}")
        features[i, :dim] = h
        coords[i, :2] = [float(v) for v in item["coord"].split("_")]
        parent = item["low_coord"]
        if parent in low:
            l = np.asarray(low[parent], dtype=np.float32)
            if l.shape != (dim,) or not np.isfinite(l).all():
                raise ValueError(f"Invalid Low feature at {parent}")
            features[i, dim:] = l
            coords[i, 2:4] = [float(v) for v in parent.split("_")]
            coords[i, 4] = 1
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite")
    return features, coords
