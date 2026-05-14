"""Label human intervention timesteps from trajectory data."""

from __future__ import annotations

import numpy as np


class InterventionLabeler:
    """Label timesteps where a human operator intervened.

    Parameters
    ----------
    action_threshold_std : float
        Number of standard deviations above which an action deviation
        is flagged as potential intervention.
    """

    def __init__(self, action_threshold_std: float = 3.0) -> None:
        self.action_threshold_std = action_threshold_std

    def label(
        self,
        observed_actions: np.ndarray,
        base_actions: np.ndarray | None = None,
        intervention_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Label per-timestep interventions.

        Parameters
        ----------
        observed_actions : np.ndarray, shape (T, action_dim)
            The recorded (possibly teleoperated) actions.
        base_actions : np.ndarray or None
            If available, actions from an autonomous policy; deviation
            from this suggests intervention.
        intervention_labels : np.ndarray or None
            Ground-truth intervention labels if available.

        Returns
        -------
        np.ndarray, shape (T,)
            Boolean intervention indicators.
        """
        T = len(observed_actions)
        interventions = np.zeros(T, dtype=bool)

        # Ground truth takes priority
        if intervention_labels is not None:
            return np.asarray(intervention_labels, dtype=bool)

        # Method 1: deviation from base policy
        if base_actions is not None and len(base_actions) == T:
            diff = np.linalg.norm(observed_actions - base_actions, axis=1)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff) + 1e-8
            for t in range(T):
                if diff[t] > mean_diff + self.action_threshold_std * std_diff:
                    interventions[t] = True
            return interventions

        # Method 2: action magnitude outlier detection
        action_mag = np.linalg.norm(observed_actions, axis=1)
        mean_mag = np.mean(action_mag)
        std_mag = np.std(action_mag) + 1e-8
        for t in range(T):
            if abs(action_mag[t] - mean_mag) > self.action_threshold_std * std_mag:
                interventions[t] = True

        return interventions
