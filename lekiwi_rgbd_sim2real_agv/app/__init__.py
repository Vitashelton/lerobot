# lekiwi_rgbd_sim2real_agv app module
#
# Demo applications and dashboards for AGV perception, navigation,
# and data collection workflows.

from .run_sim_demo import SimDemoConfig, main as run_sim_demo
from .run_real_demo import RealDemoConfig, main as run_real_demo
from .dashboard import Dashboard
from .run_offline_replay import ReplayConfig, main as run_offline_replay

__all__ = [
    "SimDemoConfig",
    "run_sim_demo",
    "RealDemoConfig",
    "run_real_demo",
    "Dashboard",
    "ReplayConfig",
    "run_offline_replay",
]
