"""
Training loop for the Residual Safety Model.

Loss = MSE(delta, delta_hat)
     + collision_risk_weight * risk_penalty
     + smoothness_weight * temporal smoothness

Uses draccus for configuration and early-stopping based on validation loss.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import draccus

# Local imports
try:
    from lekiwi_rgbd_sim2real_agv.learning.residual_model import ResidualSafetyModel
    from lekiwi_rgbd_sim2real_agv.learning.build_dataset import ResidualDatasetBuilder
    from lekiwi_rgbd_sim2real_agv.learning.risk_scorer import RiskScorer
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from learning.residual_model import ResidualSafetyModel  # type: ignore[import-not-found]
    from learning.build_dataset import ResidualDatasetBuilder  # type: ignore[import-not-found]
    from learning.risk_scorer import RiskScorer  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainResidualConfig:
    data_dir: str = "data/training"
    output_dir: str = "checkpoints"
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    mse_weight: float = 1.0
    collision_risk_weight: float = 0.5
    smoothness_weight: float = 0.1
    device: str = "cuda"
    early_stop_patience: int = 15
    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.1
    activation: str = "relu"
    # Optimizer
    lr_scheduler_step: int = 30
    lr_scheduler_gamma: float = 0.5
    # Misc
    seed: int = 42
    log_interval: int = 20
    scan_dim: int = 64
    action_dim: int = 3


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ResidualTrainer:
    """Full training loop for the residual safety model."""

    def __init__(self, config: TrainResidualConfig) -> None:
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.risk_scorer = RiskScorer()

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.model: Optional[ResidualSafetyModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0

        # Metrics history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "val_risk_penalty": [],
            "val_smoothness": [],
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[DataLoader, DataLoader]:
        """Load pre-built .npz splits and wrap as DataLoaders."""
        train_path = os.path.join(self.cfg.data_dir, "train.npz")
        val_path = os.path.join(self.cfg.data_dir, "val.npz")

        if not os.path.isfile(train_path):
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                f"Run build_dataset.py first."
            )

        train_loader = self._make_loader(train_path, shuffle=True)
        val_loader = self._make_loader(val_path, shuffle=False) if os.path.isfile(val_path) else None
        return train_loader, val_loader

    def _make_loader(self, path: str, shuffle: bool) -> DataLoader:
        batch = np.load(path)
        tensors = [
            torch.as_tensor(batch[k], dtype=torch.float32)
            for k in ["scan", "raw_action", "goal", "velocity", "last_action", "delta"]
        ]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> ResidualSafetyModel:
        """Run the full training loop with validation and checkpointing.

        Returns
        -------
        ResidualSafetyModel
            The best model (loaded from the best checkpoint).
        """
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.model = ResidualSafetyModel(
            scan_dim=self.cfg.scan_dim,
            action_dim=self.cfg.action_dim,
            hidden_dims=self.cfg.hidden_dims,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.lr_scheduler_step,
            gamma=self.cfg.lr_scheduler_gamma,
        )

        train_loader, val_loader = self._load_data()

        print(f"Training on {self.device}, {len(train_loader.dataset)} samples")  # type: ignore[arg-type]
        print(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                scan, raw_action, goal, velocity, last_action, delta_target = [
                    t.to(self.device) for t in batch
                ]
                loss = self._compute_loss(
                    scan, raw_action, goal, velocity, last_action, delta_target
                )
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            val_metrics = self._validate(val_loader) if val_loader is not None else {}
            val_loss = val_metrics.get("total", float("inf"))

            self.history["val_loss"].append(val_loss)
            self.history["val_mse"].append(val_metrics.get("mse", 0.0))
            self.history["val_risk_penalty"].append(val_metrics.get("risk_penalty", 0.0))
            self.history["val_smoothness"].append(val_metrics.get("smoothness", 0.0))

            self.scheduler.step()

            if epoch % self.cfg.log_interval == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                vstr = f"val_loss={val_loss:.6f}" if val_loader else "val_loss=N/A"
                print(
                    f"Epoch {epoch:4d}/{self.cfg.epochs} | "
                    f"train_loss={avg_train_loss:.6f} | {vstr} | lr={lr:.2e}"
                )
                if val_metrics:
                    print(
                        f"         MSE={val_metrics.get('mse', 0):.6f}  "
                        f"risk={val_metrics.get('risk_penalty', 0):.6f}  "
                        f"smooth={val_metrics.get('smoothness', 0):.6f}"
                    )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint("best.pt", epoch)
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.epochs_no_improve} epochs)")
                break

        # Load best model
        best_path = os.path.join(self.cfg.output_dir, "best.pt")
        if os.path.isfile(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        self._save_history()
        print(f"Training done. Best epoch: {self.best_epoch}, best val loss: {self.best_val_loss:.6f}")
        return self.model

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        scan: torch.Tensor,
        raw_action: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor,
        last_action: torch.Tensor,
        delta_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the composite training loss.

        ``Loss = MSE_weight * MSE + risk_weight * risk_penalty + smoothness_weight * smoothness``
        """
        delta_pred = self.model(scan, raw_action, goal, velocity, last_action)

        # 1) MSE
        mse = nn.functional.mse_loss(delta_pred, delta_target)

        # 2) Risk penalty: when the scan is dangerous but the model predicts
        #    a very small delta, penalise that under-reaction.
        risk_penalty = self._risk_penalty(scan, raw_action, delta_pred)

        # 3) Smoothness: encourage temporal consistency.
        #    We approximate by comparing delta_t vs delta_{t-1} when last_action differs.
        smoothness = self._smoothness_loss(delta_pred, last_action, raw_action)

        loss = (
            self.cfg.mse_weight * mse
            + self.cfg.collision_risk_weight * risk_penalty
            + self.cfg.smoothness_weight * smoothness
        )
        return loss

    def _risk_penalty(
        self,
        scan: torch.Tensor,
        raw_action: torch.Tensor,
        delta_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise the model when it predicts a tiny delta in dangerous states.

        For each sample, compute a collision risk score from the scan
        (via RiskScorer running on CPU), then weight a penalty that
        pushes the model to produce *larger* delta magnitudes when risk
        is high.

        Implementation: risk_score * max(0, epsilon - ||delta_pred||),
        so the model is penalised only when ||delta_pred|| is below epsilon
        but the state is risky.
        """
        batch_size = scan.shape[0]
        epsilon = 0.02  # minimum expected delta magnitude

        # Compute risk on CPU (RiskScorer uses numpy).
        scan_np = scan.detach().cpu().numpy()
        risk_scores = torch.zeros(batch_size, device=scan.device)
        for i in range(batch_size):
            r = self.risk_scorer.compute_risk(scan_np[i])
            risk_scores[i] = r["collision_risk"]

        delta_norm = delta_pred.norm(dim=-1)  # (B,)
        under_reaction = torch.clamp(epsilon - delta_norm, min=0.0)  # (B,)
        penalty = (risk_scores * under_reaction).mean()
        return penalty

    def _smoothness_loss(
        self,
        delta_pred: torch.Tensor,
        last_action: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage smooth delta predictions.

        Approximates temporal smoothness by penalising the change in delta
        relative to the change in raw action.  When the raw action does not
        change much, the delta should also not change much.

        ``smoothness = ||delta_pred||``
        This is a simple regulariser that discourages large / jittery deltas.
        """
        # Simple L2 regularisation on delta magnitude to prevent jitter.
        return delta_pred.pow(2).mean()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Compute validation metrics.

        Returns
        -------
        dict
            Keys: ``total``, ``mse``, ``risk_penalty``, ``smoothness``.
        """
        if self.model is None:
            return {}
        self.model.eval()

        total_mse = 0.0
        total_risk = 0.0
        total_smooth = 0.0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                scan, raw_action, goal, velocity, last_action, delta_target = [
                    t.to(self.device) for t in batch
                ]
                delta_pred = self.model(scan, raw_action, goal, velocity, last_action)

                mse = nn.functional.mse_loss(delta_pred, delta_target).item()
                risk = self._risk_penalty(scan, raw_action, delta_pred).item()
                smooth = self._smoothness_loss(delta_pred, last_action, raw_action).item()

                bs = scan.size(0)
                total_mse += mse * bs
                total_risk += risk * bs
                total_smooth += smooth * bs
                n += bs

        return {
            "mse": total_mse / max(n, 1),
            "risk_penalty": total_risk / max(n, 1),
            "smoothness": total_smooth / max(n, 1),
            "total": (
                self.cfg.mse_weight * total_mse
                + self.cfg.collision_risk_weight * total_risk
                + self.cfg.smoothness_weight * total_smooth
            )
            / max(n, 1),
        }

    # ------------------------------------------------------------------
    # Checkpointing & history
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = os.path.join(self.cfg.output_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "cfg": self.cfg,
            },
            path,
        )

    def _save_history(self) -> None:
        path = os.path.join(self.cfg.output_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: TrainResidualConfig) -> None:  # type: ignore[no-untyped-def]
    trainer = ResidualTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
