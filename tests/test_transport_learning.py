import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gdm.learning.data import TrajectoryDataset, file_hash
from gdm.learning.graph import collate_graphs, encode_state
from gdm.learning.model import TrajectoryRanker
from gdm.learning.train import load_policy
from gdm.transport.cli import fit
from gdm.transport.design import ACTIONS, action_features, protocol_id, read_config


def fixture_state():
    return {
        "nodes": np.array([[0., 0.], [1., 0.], [0., 1.]]),
        "edges": np.array([[0, 1], [1, 2], [2, 0]]),
        "node_mass": np.array([.2, .3, .5]),
        "field": np.array([.4, .6, .2]), "boundary": np.array([True] * 3),
        "parameters": np.array([.2, .3, .01, .1]),
        "scores": np.array([np.arange(len(ACTIONS)), np.arange(len(ACTIONS))[::-1]]) + 1,
    }


def test_training_and_checkpoint_roundtrip(tmp_path):
    cfg = read_config(Path(__file__).resolve().parents[1] / "config" / "transport.yaml")
    cfg["learning"].update(epochs=2, latent_dim=8, message_passing_steps=1)
    state = fixture_state()
    samples = []
    for split in ("train", "validation"):
        sample = tmp_path / f"{split}.npz"
        np.savez_compressed(sample, **state)
        samples.append({"split": split, "case_id": split, "window": 0, "file": sample.name,
                        "sha256": file_hash(sample)})
    manifest = {"schema": 2, "protocol": protocol_id(cfg), "actions": action_features().tolist(),
                "samples": samples}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    for target in ("instantaneous", "window"):
        out = tmp_path / target
        fit(out, tmp_path, cfg, target, "cpu", 27)
        policy, metadata = load_policy(out / "best.pt", protocol=protocol_id(cfg), actions=action_features())
        assert metadata["target"] == target and metadata["seed"] == 27
        choice = policy.action(state)
        assert 0 <= choice < len(ACTIONS)
        if target == "instantaneous":
            assert ACTIONS[choice].timing == 0
        assert json.loads((out / "training.json").read_text())["status"] == "complete"
    manifest["samples"][1]["case_id"] = "train"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        TrajectoryDataset(tmp_path, protocol_id(cfg), action_features(), "validation", "window")


def test_candidate_scores_are_invariant_to_node_index_and_batching():
    state = fixture_state()
    order = np.array([2, 0, 1])
    permuted = {**state}
    for name in ("nodes", "node_mass", "field", "boundary"):
        permuted[name] = state[name][order]
    permuted["edges"] = np.argsort(order)[state["edges"]]
    first, second = encode_state(state), encode_state(permuted)
    actions = torch.as_tensor(action_features(), dtype=torch.float32)
    model = TrajectoryRanker(latent_dim=8, message_passing_steps=2, action_dimension=2)
    with torch.inference_mode():
        pair = model(collate_graphs([first, second]), actions)
        one = model(collate_graphs([first]), actions)
        inverse = torch.arange(len(actions) - 1, -1, -1)
        reordered = model(collate_graphs([first]), actions[inverse])
    torch.testing.assert_close(pair[0], pair[1])
    torch.testing.assert_close(pair[:1], one)
    torch.testing.assert_close(one[:, inverse], reordered)
