"""LeKiwi real-robot deployment adapter.

Connects the offline-trained RL policy to the physical LeKiwi robot
via ZMQ communication, D435i RGB-D camera, and safety filtering.
"""

from lekiwi_deployment.observation_assembler import ObservationAssembler
from lekiwi_deployment.deployment_runner import DeploymentRunner

__all__ = ["ObservationAssembler", "DeploymentRunner"]
