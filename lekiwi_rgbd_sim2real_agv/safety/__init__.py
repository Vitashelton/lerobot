"""Rule-based safety filter for real-robot deployment.

Layers (in order):
    1. Invalid depth fallback → zero velocity
    2. Communication timeout fallback → zero velocity
    3. Emergency stop (front < d_stop) → vx = 0
    4. Lateral inhibit (side < d_lat) → vy = 0
    5. Velocity scaling proportional to clearance
    6. Rotation stop (any < d_rot) → omega = 0
    7. Action clipping to limits
    8. Acceleration limiting
    9. Low-pass temporal smoothing
"""

from safety.safety_filter import SafetyFilter

__all__ = ["SafetyFilter"]
