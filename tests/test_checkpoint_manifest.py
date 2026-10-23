"""The checkpoint manifest tool is checked.

The manifest is what goes into git instead of ~100 MB of float32 weights, so
it has to carry the facts a reader needs to tie a systems number to a stated
quality level: which arm, which seed, how many steps, what held-out loss, how
many bytes, and the SHA-256 of the file it stands for.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_manifest_lists_every_checkpoint_with_its_facts(tmp_path):
    ckdir = tmp_path / "checkpoints"
    ckdir.mkdir()
    payloads = {
        "HE_seed1.pt": {"arm": "HE", "seed": 1, "steps": 2000,
                        "final_held_out_loss": 6.632957458496094,
                        "config": {"d": 128}, "evals": [], "state_dict": {}},
        "attention-32h_seed2.pt": {"arm": "attention-32h", "seed": 2, "steps": 2000,
                                   "final_held_out_loss": 6.596700310707092,
                                   "config": {"d": 128}, "evals": [], "state_dict": {}},
    }
    for name, payload in payloads.items():
        torch.save(payload, ckdir / name)

    out = tmp_path / "MANIFEST.json"
    r = subprocess.run([PY, str(REPO / "tools" / "make_checkpoint_manifest.py"),
                        "--dir", str(ckdir), "--out", str(out), "--label", "full"],
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["schema"] == "untransformers-checkpoint-manifest/1"
    assert manifest["label"] == "full"
    entries = {e["file"]: e for e in manifest["checkpoints"]}
    assert set(entries) == set(payloads)
    for name, payload in payloads.items():
        e = entries[name]
        assert e["arm"] == payload["arm"]
        assert e["seed"] == payload["seed"]
        assert e["steps"] == payload["steps"]
        assert e["final_held_out_loss"] == payload["final_held_out_loss"]
        assert e["bytes"] == (ckdir / name).stat().st_size
        assert e["sha256"] == _sha256(ckdir / name)


def test_manifest_survives_an_unreadable_checkpoint(tmp_path):
    ckdir = tmp_path / "checkpoints"
    ckdir.mkdir()
    (ckdir / "garbage.pt").write_bytes(b"not a torch archive")

    out = tmp_path / "MANIFEST.json"
    r = subprocess.run([PY, str(REPO / "tools" / "make_checkpoint_manifest.py"),
                        "--dir", str(ckdir), "--out", str(out)],
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads(out.read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    assert entry["file"] == "garbage.pt"
    assert entry["sha256"] == _sha256(ckdir / "garbage.pt")
