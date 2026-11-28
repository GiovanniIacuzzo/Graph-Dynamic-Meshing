"""Portable dataset contract shared by numerical producers and graph learners."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import collate_graphs, encode_state


TARGETS = {"instantaneous": 0, "window": 1}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryDataset(Dataset):
    def __init__(self, directory, protocol, actions, split, target):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 2 or manifest["protocol"] != protocol:
            raise ValueError("Dataset schema or protocol mismatch.")
        if not np.array_equal(manifest["actions"], actions):
            raise ValueError("Dataset action descriptors differ from the requested actions.")
        valid = {"train", "validation"}
        assignments = {}
        seen = set()
        files = set()
        samples = []
        for row in manifest["samples"]:
            case, current = row["case_id"], row["split"]
            key = (case, row["window"])
            if (current not in valid or assignments.setdefault(case, current) != current
                    or key in seen or row["file"] in files):
                raise ValueError("Duplicate sample, overlapping split, or test data in training manifest.")
            seen.add(key)
            files.add(row["file"])
            if current == split:
                samples.append(row)
        if not samples:
            raise ValueError(f"Empty dataset split: {split}.")
        self.items = []
        for row in samples:
            path = (self.directory / row["file"]).resolve()
            if path.parent != self.directory or file_hash(path) != row["sha256"]:
                raise ValueError(f"Invalid sample path or checksum: {row['file']}")
            with np.load(path, allow_pickle=False) as data:
                scores = data["scores"]
                if scores.shape != (len(TARGETS), len(actions)) or not np.isfinite(scores).all() or np.any(scores < 0):
                    raise ValueError(f"Invalid reference scores: {row['file']}")
                self.items.append((encode_state(data), scores[TARGETS[target]].astype(np.float32),
                                   row["case_id"], row["window"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_samples(items):
    graphs, scores, cases, windows = zip(*items)
    return collate_graphs(graphs), torch.as_tensor(np.stack(scores)), cases, windows
