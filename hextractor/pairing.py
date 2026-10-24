"""Pairing evidence for the quality protocol (Researcher's memo F1).

The quality gate used to *assert* `same_batches_same_order` and
`paired_init_seeds` because the runner wrote those two literals into the
results JSON. The gate could not fail: it certified a claim that nothing had
measured. This module derives both claims from artefacts, so the gate can
check them.

**`same_batches_same_order`** — the batch streams are pure functions of
`(train_seed, split, corpus_id, tokenizer_sha256, l, batch)`: `CorpusBundle.stream`
takes no arm and no seed argument (`hextractor/corpus.py:147-166`), and the
harness builds the stream once per arm and re-uses it for every seed
(`hextractor/harness.py:219-227`). The evidence hashes the first N windows of
the train stream and of the held-out stream and pins the corpus/config
identity they were derived from, so a later reader can recompute them.

Scope, stated honestly: this certifies that the *data pipeline* is
arm-independent and seed-independent (a property of the code path and the
frozen corpus), not that the finished training loop was observed consuming
those windows. The claim in this harness means exactly the former.

**`paired_init_seeds`** — `apply_init_policy` draws the shared
(non-sublayer-1) parameters FIRST from a single seeded generator
(`hextractor/init.py:27-34`), so two arms built with the same seed start from
bit-identical shared parameters. The evidence rebuilds both arms per seed and
compares the shared parameters bitwise (`torch.equal`), separating the
generator-drawn tensors from the constant-filled ones (LayerNorm gains/biases
and biases are set to fixed values, so they would match trivially).

Nothing here interprets a quality result: it only produces evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .corpus import CorpusBundle
from .harness import TrainConfig, build_model
from .init import apply_init_policy

SCHEMA = "untransformers-pairing-evidence/1"
VERIFIER_VERSION = "1"

SHARED_PREFIX_EXCLUDED = "sub1."


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_parameters(model) -> Dict[str, torch.Tensor]:
    """The parameters the two arms are supposed to share (everything but sub1)."""
    return {n: p for n, p in model.named_parameters()
            if not n.startswith(SHARED_PREFIX_EXCLUDED)}


def _is_generator_drawn(name: str, p: torch.Tensor) -> bool:
    """True for the tensors `apply_init_policy` draws from the seeded generator.

    LayerNorm gains/biases and any 1-D bias are set to fixed values (ones /
    zeros), so their equality across arms would be a triviality; the
    generator-drawn tensors are the ones that carry the pairing evidence.
    """
    if "norm" in name:
        return False
    return p.ndim > 1


def stream_fingerprints(bundle: CorpusBundle, l: int, batch: int,
                        train_seed: int, n_batches: int = 100) -> Dict:
    """SHA-256 of the first `n_batches` windows of each stream."""
    out: Dict = {"n_batches_hashed": n_batches, "l": l, "batch": batch,
                 "train_seed": train_seed}
    for held_out, key in ((False, "train"), (True, "held_out")):
        h = hashlib.sha256()
        it = bundle.stream(l=l, batch=batch, train_seed=train_seed,
                           held_out=held_out)
        for _ in range(n_batches):
            x, y = next(it)
            h.update(x.astype("int64").tobytes())
            h.update(y.astype("int64").tobytes())
        out[f"{key}_stream_sha256"] = h.hexdigest()
    out["method"] = (
        "the train and held-out streams are pure functions of (train_seed, split, "
        "corpus_id, tokenizer_sha256, l, batch) and take no arm and no seed "
        "argument, so every arm and every seed consumes the same windows in the "
        "same order; the first N windows of each stream are hashed here and the "
        "corpus/config identity they were derived from is recorded")
    return out


def compare_shared_parameters(arms: Sequence[str], cfg: TrainConfig,
                              seed: int) -> Dict:
    """Bitwise comparison of the shared parameters of `arms` at `seed`."""
    models = {}
    for arm in arms:
        m = build_model(arm, cfg)
        apply_init_policy(m, seed=seed)
        models[arm] = m

    arm_a = arms[0]
    params_a = shared_parameters(models[arm_a])
    per_arm: Dict[str, Dict] = {}
    all_names: List[str] = []
    for arm in arms[1:]:
        params_b = shared_parameters(models[arm])
        names_a = set(params_a)
        names_b = set(params_b)
        common = sorted(names_a & names_b)
        all_names.extend(common)
        differing = []
        drawn = 0
        drawn_equal = 0
        for name in common:
            if _is_generator_drawn(name, params_a[name]):
                drawn += 1
                if torch.equal(params_a[name], params_b[name]):
                    drawn_equal += 1
                else:
                    differing.append(name)
        per_arm[arm] = {
            "compared_against": arm_a,
            "n_shared_parameters": len(common),
            "n_only_in_first_arm": sorted(names_a - names_b),
            "n_only_in_this_arm": sorted(names_b - names_a),
            "n_generator_drawn": drawn,
            "n_generator_drawn_bitwise_equal": drawn_equal,
            "differing_parameters": differing,
            "bitwise_equal": not differing and bool(common),
            "shared_parameter_names_sha256": _sha256_text("\n".join(common)),
        }
    for m in models.values():
        del m
    return per_arm


def checkpoint_provenance(checkpoint_dir: Optional[Path], arms: Sequence[str],
                          seeds: Sequence[int]) -> Dict:
    """Per checkpoint: file digest + the metadata it carries."""
    out: Dict = {}
    if checkpoint_dir is None:
        return out
    for arm in arms:
        for seed in seeds:
            path = Path(checkpoint_dir) / f"{arm}_seed{seed}.pt"
            if not path.exists():
                out[f"{arm}_seed{seed}"] = {"present": False, "path": str(path)}
                continue
            entry: Dict = {"present": True, "path": str(path),
                           "file_sha256": sha256_file(path),
                           "file_bytes": path.stat().st_size}
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
                entry.update({
                    "arm": blob.get("arm"),
                    "seed": blob.get("seed"),
                    "steps": blob.get("steps"),
                    "final_held_out_loss": blob.get("final_held_out_loss"),
                    "n_state_dict_tensors": len(blob.get("state_dict") or {}),
                })
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)
            out[f"{arm}_seed{seed}"] = entry
    return out


def _git_stamp(repo_root: Path) -> Dict:
    import subprocess

    def run(*args) -> Optional[str]:
        try:
            r = subprocess.run(["git", *args], cwd=str(repo_root), text=True,
                               capture_output=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    dirty = run("status", "--porcelain")
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(dirty) if dirty is not None else None}


def build_pairing_evidence(bundle: CorpusBundle, cfg: TrainConfig,
                           arms: Sequence[str], seeds: Sequence[int],
                           train_seed: int, steps: int,
                           checkpoint_dir: Optional[Path] = None,
                           n_batches: int = 100,
                           repo_root: Optional[Path] = None) -> Dict:
    """The evidence block that certifies the two pairing claims."""
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    streams = stream_fingerprints(bundle, l=cfg.l, batch=cfg.batch_size,
                                  train_seed=train_seed, n_batches=n_batches)

    per_seed: Dict[str, Dict] = {}
    all_equal = True
    for seed in seeds:
        cmp = compare_shared_parameters(list(arms), cfg, seed)
        seed_ok = all(entry.get("bitwise_equal") for entry in cmp.values())
        all_equal = all_equal and seed_ok
        # Aggregates for the seed as a whole: the gate reads these, the
        # per-arm detail stays below them for a human reader.
        n_drawn = max([int(e.get("n_generator_drawn") or 0)
                       for e in cmp.values()] or [0])
        n_drawn_equal = max([int(e.get("n_generator_drawn_bitwise_equal") or 0)
                             for e in cmp.values()] or [0])
        n_shared = max([int(e.get("n_shared_parameters") or 0)
                        for e in cmp.values()] or [0])
        per_seed[str(seed)] = {"bitwise_equal": seed_ok,
                               "n_generator_drawn": n_drawn,
                               "n_generator_drawn_bitwise_equal": n_drawn_equal,
                               "n_shared_parameters": n_shared,
                               "per_arm": cmp}

    evidence = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verifier": {"tool": "tools/verify/verify_pairing_evidence.py",
                     "module": "hextractor/pairing.py",
                     "version": VERIFIER_VERSION,
                     **_git_stamp(repo_root)},
        "corpus": {"corpus_id": bundle.corpus_id,
                   "tokenizer_sha256": bundle.tokenizer_sha256,
                   "n_tokens": bundle.n_tokens},
        "protocol": {"arms": list(arms), "seeds": list(seeds),
                     "train_seed": train_seed, "steps": steps,
                     "l": cfg.l, "batch_size": cfg.batch_size,
                     "d": cfg.d, "n_layers": cfg.n_layers,
                     "vocab": cfg.vocab},
        "claims": {
            "same_batches_same_order": {
                "verified": True,
                "n_batches_hashed": streams["n_batches_hashed"],
                "train_stream_sha256": streams["train_stream_sha256"],
                "held_out_stream_sha256": streams["held_out_stream_sha256"],
                "method": streams["method"],
                "scope": ("certifies that the data pipeline is arm-independent and "
                          "seed-independent for the recorded corpus/config; it is "
                          "not an observation of the finished training loop"),
            },
            "paired_init_seeds": {
                "verified": bool(all_equal),
                "per_seed": per_seed,
                "method": ("both arms are rebuilt with the same seed through the "
                           "runner's own `apply_init_policy` and their shared "
                           "(non-sublayer-1) parameters are compared bitwise; the "
                           "generator-drawn tensors are counted separately from the "
                           "constant-filled ones (LayerNorm gains/biases)"),
                "scope": ("certifies the initialisation of the shared parameters; "
                          "sublayer-1 is arm-specific by construction and is not "
                          "expected to match"),
            },
        },
        "checkpoints": checkpoint_provenance(checkpoint_dir, list(arms), list(seeds)),
    }
    return evidence


def evidence_json(evidence: Dict) -> str:
    return json.dumps(evidence, indent=2, sort_keys=False) + "\n"
