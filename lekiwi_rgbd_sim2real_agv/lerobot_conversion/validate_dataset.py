"""Validate a LeRobot dataset for integrity and schema compliance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch


def validate_lerobot_dataset(data_dir: str) -> Dict[str, any]:
    """Check dataset integrity and return a validation report.

    Parameters
    ----------
    data_dir : str
        Path to the LeRobot dataset directory.

    Returns
    -------
    dict
        Validation report with keys:
        - ``valid``: bool
        - ``errors``: list of str
        - ``warnings``: list of str
        - ``stats``: dict with counts, shapes, etc.
    """
    from lerobot_conversion.unified_to_lerobot import UnifiedToLeRobotConverter

    report: dict = {"valid": True, "errors": [], "warnings": [], "stats": {}}

    data_path = Path(data_dir)
    if not data_path.exists():
        report["valid"] = False
        report["errors"].append(f"Data directory does not exist: {data_dir}")
        return report

    # Check required files
    for fname in ["info.json", "stats.json"]:
        if not (data_path / fname).exists():
            report["warnings"].append(f"Missing {fname}")

    # Check data chunks
    data_subdir = data_path / "data"
    if not data_subdir.exists():
        report["valid"] = False
        report["errors"].append("No data/ directory found")
        return report

    chunks = sorted(data_subdir.glob("chunk-*.pt"))
    if not chunks:
        report["valid"] = False
        report["errors"].append("No data chunks (chunk-*.pt) found")
        return report

    # Validate chunk consistency
    try:
        first_chunk = torch.load(chunks[0], weights_only=False)
        expected_keys = set(first_chunk.keys())
        expected_shapes = {
            k: list(v.shape[1:]) for k, v in first_chunk.items()
        }
        total_frames = 0
        for cp in chunks:
            chunk = torch.load(cp, weights_only=False)
            if set(chunk.keys()) != expected_keys:
                report["errors"].append(
                    f"Key mismatch in {cp.name}: got {set(chunk.keys())}, "
                    f"expected {expected_keys}"
                )
                report["valid"] = False
            for k, v in chunk.items():
                if list(v.shape[1:]) != expected_shapes.get(k):
                    report["errors"].append(
                        f"Shape mismatch in {cp.name}[{k}]: "
                        f"got {list(v.shape[1:])}, expected {expected_shapes.get(k)}"
                    )
                    report["valid"] = False
            total_frames += len(next(iter(chunk.values())))

        report["stats"] = {
            "num_chunks": len(chunks),
            "total_frames": total_frames,
            "keys": sorted(expected_keys),
            "shapes": expected_shapes,
        }
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to load chunks: {e}")

    # Check required observation keys
    required_keys = {"action", "observation.images.rgb"}
    missing = required_keys - expected_keys
    if missing:
        report["warnings"].append(f"Missing recommended keys: {missing}")

    return report
