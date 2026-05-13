"""
Build training dataset from synthetic + real data for the residual safety model.

Converts raw logs (synthetic episodes or real LeKiwi teleop logs) into
(scan, action, goal, velocity) tuples, generates weak labels via DWA safety
projection, applies augmentation, and splits into train / val / test .npz files.
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

# Local imports (used at runtime when the package is installed).
try:
    from lekiwi_rgbd_sim2real_agv.control.dwa_policy import DWAPolicy
except ImportError:
    # Fallback when running from the source tree without installation.
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from control.dwa_policy import DWAPolicy  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BuildDatasetConfig:
    sim_data_dir: str = "data/synthetic"
    real_data_dir: str = "data/real_logs"
    output_dir: str = "data/training"
    train_split: float = 0.7
    val_split: float = 0.15
    use_augmentation: bool = True

    # Scan settings
    scan_dim: int = 64
    scan_max_range: float = 5.0

    # DWA label generation
    dwa_vx_samples: int = 7
    dwa_vy_samples: int = 7
    dwa_omega_samples: int = 15

    # Augmentation
    augment_sigma_scan: float = 0.02  # metres std
    augment_sigma_action: float = 0.01  # m/s or rad/s std
    augment_mirror_prob: float = 0.3


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ResidualDatasetBuilder:
    """Build training dataset from synthetic + real data.

    The builder:
    1. Loads synthetic episodes (JSON / .npz with poses, scans, goals).
    2. Loads real LeKiwi logs (scan + action pairs).
    3. Generates delta-action labels via DWA safety projection.
    4. (optionally) augments data with noise and mirroring.
    5. Splits and writes train / val / test .npz archives.
    """

    def __init__(self, config: BuildDatasetConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Build entry point
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """Main entry point: load, label, augment, split, save.

        Returns
        -------
        dict
            ``{"train": list, "val": list, "test": list}`` of
            ``(features, labels)`` tuples.
        """
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        print("Loading synthetic data ...")
        sim_data = self._load_sim_data()

        print("Loading real data ...")
        real_data = self._load_real_data()

        all_data = sim_data + real_data
        print(f"Total raw samples: {len(all_data)} (sim={len(sim_data)}, real={len(real_data)})")

        print("Generating labels via DWA ...")
        labeled = self._generate_labels(all_data)

        if self.cfg.use_augmentation:
            print("Augmenting ...")
            labeled = self._augment(labeled)

        # Shuffle
        indices = self.rng.permutation(len(labeled))
        labeled = [labeled[i] for i in indices]

        # Split
        n = len(labeled)
        n_train = int(n * self.cfg.train_split)
        n_val = int(n * self.cfg.val_split)

        train = labeled[:n_train]
        val = labeled[n_train : n_train + n_val]
        test = labeled[n_train + n_val :]

        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        # Save
        self._save_split(train, "train")
        self._save_split(val, "val")
        self._save_split(test, "test")

        return {"train": train, "val": val, "test": test}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_sim_data(self) -> list[dict[str, np.ndarray]]:
        """Load synthetic scenes, simulating scans at various poses."""
        data: list[dict[str, np.ndarray]] = []
        sim_dir = self.cfg.sim_data_dir
        if not os.path.isdir(sim_dir):
            print(f"  [WARN] sim data dir not found: {sim_dir}")
            return data

        # Try .npz first, then .json
        for fname in sorted(glob.glob(os.path.join(sim_dir, "*.npz"))):
            try:
                batch = dict(np.load(fname, allow_pickle=True))
                data.extend(self._parse_sim_batch(batch))
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")

        for fname in sorted(glob.glob(os.path.join(sim_dir, "*.json"))):
            try:
                with open(fname) as f:
                    entries = json.load(f)
                if isinstance(entries, dict):
                    entries = [entries]
                for entry in entries:
                    sample = self._parse_sim_sample(entry)
                    if sample is not None:
                        data.append(sample)
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")

        return data

    def _load_real_data(self) -> list[dict[str, np.ndarray]]:
        """Load real LeKiwi logs, extracting scan + action pairs."""
        data: list[dict[str, np.ndarray]] = []
        real_dir = self.cfg.real_data_dir
        if not os.path.isdir(real_dir):
            print(f"  [WARN] real data dir not found: {real_dir}")
            return data

        for fname in sorted(glob.glob(os.path.join(real_dir, "*.npz"))):
            try:
                batch = dict(np.load(fname, allow_pickle=True))
                data.extend(self._parse_real_batch(batch))
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")

        for fname in sorted(glob.glob(os.path.join(real_dir, "*.json"))):
            try:
                with open(fname) as f:
                    entries = json.load(f)
                if isinstance(entries, dict):
                    entries = [entries]
                for entry in entries:
                    sample = self._parse_real_sample(entry)
                    if sample is not None:
                        data.append(sample)
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")

        return data

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_sim_batch(self, batch: dict) -> list[dict[str, np.ndarray]]:
        """Parse a .npz batch of synthetic data.

        Expected keys (each shape (N, ...)):
            scans, raw_actions, goals, velocities, last_actions (optional)
        """
        items: list[dict[str, np.ndarray]] = []
        scans = batch.get("scans", batch.get("scan"))
        actions = batch.get("raw_actions", batch.get("raw_action", batch.get("actions")))
        goals = batch.get("goals", batch.get("goal"))
        velocities = batch.get("velocities", batch.get("velocity"))
        last_actions = batch.get("last_actions", batch.get("last_action"))

        if scans is None or actions is None:
            return items

        n = len(scans)
        for i in range(n):
            sample: dict[str, np.ndarray] = {
                "scan": np.asarray(scans[i], dtype=np.float32),
                "raw_action": np.asarray(actions[i], dtype=np.float32),
                "goal": (
                    np.asarray(goals[i], dtype=np.float32)
                    if goals is not None
                    else np.zeros(3, dtype=np.float32)
                ),
                "velocity": (
                    np.asarray(velocities[i], dtype=np.float32)
                    if velocities is not None
                    else np.zeros(3, dtype=np.float32)
                ),
            }
            if last_actions is not None and i < len(last_actions):
                sample["last_action"] = np.asarray(last_actions[i], dtype=np.float32)
            items.append(sample)
        return items

    def _parse_real_batch(self, batch: dict) -> list[dict[str, np.ndarray]]:
        """Same structure as sim batch but from real logs."""
        return self._parse_sim_batch(batch)

    def _parse_sim_sample(self, entry: dict) -> Optional[dict[str, np.ndarray]]:
        """Parse a single JSON sample."""
        required = {"scan", "raw_action"}
        if not required.issubset(entry):
            return None
        return {
            "scan": np.asarray(entry["scan"], dtype=np.float32),
            "raw_action": np.asarray(entry["raw_action"], dtype=np.float32),
            "goal": np.asarray(entry.get("goal", [0, 0, 0]), dtype=np.float32),
            "velocity": np.asarray(entry.get("velocity", [0, 0, 0]), dtype=np.float32),
            "last_action": np.asarray(entry.get("last_action", entry["raw_action"]), dtype=np.float32),
        }

    def _parse_real_sample(self, entry: dict) -> Optional[dict[str, np.ndarray]]:
        """Parse a single JSON sample from real logs."""
        # Real data may store the action under "action" instead of "raw_action".
        if "action" in entry and "raw_action" not in entry:
            entry = {**entry, "raw_action": entry["action"]}
        return self._parse_sim_sample(entry)

    # ------------------------------------------------------------------
    # Label generation via DWA
    # ------------------------------------------------------------------

    def _generate_labels(self, data: list[dict[str, np.ndarray]]) -> list[dict]:
        """Generate delta_action labels using DWA as the expert policy.

        For each sample:
          1. Run DWA policy to compute a safe action.
          2. delta = safe_action - raw_action, clamped.
        """
        dwa = DWAPolicy(
            vx_samples=self.cfg.dwa_vx_samples,
            vy_samples=self.cfg.dwa_vy_samples,
            omega_samples=self.cfg.dwa_omega_samples,
        )

        labeled: list[dict] = []
        for sample in tqdm(data, desc="Labelling"):
            try:
                safe = dwa.compute_action(
                    scan_m=sample["scan"],
                    goal_position=sample["goal"],
                    current_velocity=sample["velocity"],
                )
                safe_action = np.array(
                    [safe["x.vel"], safe["y.vel"], safe["theta.vel"]],
                    dtype=np.float32,
                )
                raw = sample["raw_action"].astype(np.float32)
                delta = safe_action - raw

                # Clamp delta to reasonable range
                max_delta = np.array([0.5, 0.5, 90.0], dtype=np.float32)
                delta = np.clip(delta, -max_delta, max_delta)

                labeled.append({
                    "scan": sample["scan"],
                    "raw_action": raw,
                    "goal": sample["goal"],
                    "velocity": sample["velocity"],
                    "last_action": sample.get("last_action", raw.copy()),
                    "delta": delta,
                })
            except Exception as e:
                # Skip samples where DWA fails (e.g. NaN scan).
                print(f"  [SKIP] DWA labelling failed: {e}")

        return labeled

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, data: list[dict]) -> list[dict]:
        """Apply noise to scan, perturb action, mirror left-right."""
        augmented = list(data)  # copy original
        for sample in tqdm(data, desc="Augmenting"):
            # Gaussian noise on scan
            noisy = dict(sample)  # shallow copy
            noise = self.rng.normal(0, self.cfg.augment_sigma_scan, size=noisy["scan"].shape)
            noisy["scan"] = np.clip(noisy["scan"] + noise.astype(np.float32), 0.01, self.cfg.scan_max_range)
            noisy["raw_action"] = noisy["raw_action"] + self.rng.normal(
                0, self.cfg.augment_sigma_action, size=noisy["raw_action"].shape
            ).astype(np.float32)
            noisy["last_action"] = noisy["last_action"] + self.rng.normal(
                0, self.cfg.augment_sigma_action, size=noisy["last_action"].shape
            ).astype(np.float32)
            augmented.append(noisy)

            # Mirror (left-right swap)
            if self.rng.random() < self.cfg.augment_mirror_prob:
                mirrored = dict(sample)
                # Reverse scan order (simulates mirror)
                mirrored["scan"] = mirrored["scan"][::-1].copy()
                # Swap vy sign
                mirrored["raw_action"] = mirrored["raw_action"].copy()
                mirrored["raw_action"][1] *= -1.0
                mirrored["goal"] = mirrored["goal"].copy()
                mirrored["goal"][1] *= -1.0
                mirrored["velocity"] = mirrored["velocity"].copy()
                mirrored["velocity"][1] *= -1.0
                la = mirrored.get("last_action")
                if la is not None:
                    mirrored["last_action"] = la.copy()
                    mirrored["last_action"][1] *= -1.0
                if "delta" in mirrored:
                    mirrored["delta"] = mirrored["delta"].copy()
                    mirrored["delta"][1] *= -1.0
                augmented.append(mirrored)

        return augmented

    # ------------------------------------------------------------------
    # Save / load helpers
    # ------------------------------------------------------------------

    def _save_split(self, data: list[dict], name: str) -> None:
        """Save a split to ``output_dir/{name}.npz``."""
        if not data:
            print(f"  [SKIP] No data for split '{name}'")
            return

        # Collate into arrays
        keys = ["scan", "raw_action", "goal", "velocity", "last_action", "delta"]
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            arrays[key] = np.stack([d[key] for d in data], axis=0)

        path = os.path.join(self.cfg.output_dir, f"{name}.npz")
        np.savez_compressed(path, **arrays)
        print(f"  Saved {len(data)} samples -> {path}")

    @staticmethod
    def load_split(path: str) -> list[dict[str, np.ndarray]]:
        """Load a saved split back into list-of-dicts form."""
        batch = np.load(path, allow_pickle=True)
        n = len(batch["scan"])
        keys = [k for k in batch.files]
        out: list[dict[str, np.ndarray]] = []
        for i in range(n):
            sample = {}
            for k in keys:
                arr = batch[k][i]
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                sample[k] = np.asarray(arr, dtype=np.float32)
            out.append(sample)
        return out
