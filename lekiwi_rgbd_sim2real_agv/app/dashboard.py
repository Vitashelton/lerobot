"""Flask-based web dashboard for real-time monitoring."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Dashboard:
    """Web dashboard showing:

    - RGB with detection overlays
    - Depth colormap
    - Scan curve (Chart.js or SVG)
    - Top-down safety map
    - FPS, latency, safety state
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port

        # Thread-safe shared state
        self.latest: dict[str, Any] = {}
        self.lock = threading.Lock()

        # Lazy-created Flask app
        self._app = None

        # History buffers for scan and fps
        self._scan_history: list[np.ndarray] = []
        self._fps_history: list[float] = []
        self._max_history = 300

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: dict) -> None:
        """Update latest data (called from perception loop). Thread-safe."""
        with self.lock:
            self.latest = data.copy()

        # Update history buffers
        scan = data.get("scan")
        if scan is not None and len(scan) > 0:
            self._scan_history.append(np.asarray(scan, dtype=np.float32))
            if len(self._scan_history) > self._max_history:
                self._scan_history = self._scan_history[-self._max_history:]

        fps_val = data.get("fps", 0.0)
        self._fps_history.append(float(fps_val))
        if len(self._fps_history) > self._max_history:
            self._fps_history = self._fps_history[-self._max_history:]

    def run(self) -> None:
        """Start the Flask development server (blocking)."""
        try:
            from flask import Flask
        except ImportError:
            logger.error("Flask is not installed. Install with: pip install flask")
            return

        self._app = Flask(__name__)
        self._setup_routes()
        logger.info("Dashboard starting at http://%s:%d", self.host, self.port)
        self._app.run(host=self.host, port=self.port, threaded=True)

    def run_threaded(self) -> threading.Thread:
        """Run the dashboard in a background thread. Returns the thread."""
        t = threading.Thread(target=self.run, daemon=True, name="dashboard")
        t.start()
        return t

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:
        # Bound methods for Flask routes
        app = self._app

        @app.route("/")
        def index():
            return self._html_page()

        @app.route("/api/state")
        def api_state():
            with self.lock:
                state = {
                    "risk_level": self.latest.get("risk_level", "unknown"),
                    "front_min": self.latest.get("front_min", 0),
                    "left_min": self.latest.get("left_min", 0),
                    "right_min": self.latest.get("right_min", 0),
                    "pallet_detected": self.latest.get("pallet_detected", False),
                    "fps": self.latest.get("fps", 0),
                    "latency_ms": self.latest.get("latency_ms", 0),
                    "shield_active": self.latest.get("shield_active", False),
                    "step": self.latest.get("step", 0),
                    "timestamp": time.time(),
                }
            return jsonify(state)

        @app.route("/api/scan")
        def api_scan():
            """Return the latest scan data for Chart.js rendering."""
            with self.lock:
                scan = self.latest.get("scan")
                if scan is not None and len(scan) > 0:
                    scan_list = np.nan_to_num(np.asarray(scan), nan=0.0).tolist()
                else:
                    scan_list = []
            return jsonify({
                "scan": scan_list,
                "scan_dim": len(scan_list),
                "front_min": self.latest.get("front_min", 0),
                "left_min": self.latest.get("left_min", 0),
                "right_min": self.latest.get("right_min", 0),
            })

        @app.route("/video/rgb")
        def video_rgb():
            from flask import Response
            return Response(
                self._generate_rgb_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video/depth")
        def video_depth():
            from flask import Response
            return Response(
                self._generate_depth_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    # ------------------------------------------------------------------
    # MJPEG stream generators
    # ------------------------------------------------------------------

    def _generate_rgb_stream(self):
        """MJPEG stream of RGB image with perception overlays."""
        while True:
            with self.lock:
                frame = self.latest.get("rgb_overlay")  # pre-rendered overlay
                if frame is None:
                    frame = self.latest.get("rgb")
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
            time.sleep(0.05)

    def _generate_depth_stream(self):
        """MJPEG stream of depth colormap."""
        while True:
            with self.lock:
                depth = self.latest.get("depth")
            if depth is not None:
                depth_vis = self._depth_to_colormap(depth)
                _, jpeg = cv2.imencode(".jpg", depth_vis,
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
            time.sleep(0.05)

    @staticmethod
    def _depth_to_colormap(depth_m: np.ndarray) -> np.ndarray:
        """Convert depth in meters to a colormapped BGR image."""
        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.001)]
        if len(valid) > 0:
            lo, hi = np.percentile(valid, [2, 98])
        else:
            lo, hi = 0.0, 5.0
        hi = max(hi, lo + 0.1)
        norm = (depth_m - lo) / (hi - lo)
        norm = np.clip(np.nan_to_num(norm, nan=0.0), 0, 1)
        colormap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return colormap

    # ------------------------------------------------------------------
    # HTML page
    # ------------------------------------------------------------------

    def _html_page(self) -> str:
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeKiwi AGV Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; }
        .header { background: #16213e; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.3em; color: #0ff; }
        .container { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 10px; padding: 10px; max-width: 1400px; margin: 0 auto; }
        .panel { background: #0f3460; border-radius: 8px; padding: 8px; overflow: hidden; }
        .panel h3 { font-size: 0.9em; margin-bottom: 6px; color: #e94560; }
        .video-panel { grid-row: span 1; }
        .video-panel img { width: 100%; border-radius: 4px; }
        .scan-panel { grid-column: span 2; }
        .status-bar { background: #16213e; padding: 10px 20px; display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.85em; }
        .status-item { display: flex; align-items: center; gap: 6px; }
        .status-label { color: #888; }
        .status-value { font-weight: bold; }
        .status-value.safe { color: #0f0; }
        .status-value.warning { color: #fc0; }
        .status-value.danger { color: #f00; }
        .shield-active { animation: pulse 0.5s infinite alternate; }
        @keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>LeKiwi AGV Dashboard</h1>
        <span id="connection-status" style="color:#888;">Connecting...</span>
    </div>

    <div class="container">
        <div class="panel video-panel">
            <h3>RGB Camera</h3>
            <img src="/video/rgb" id="rgb-feed" alt="RGB Stream">
        </div>
        <div class="panel">
            <h3>Depth Colormap</h3>
            <img src="/video/depth" style="width:100%;" alt="Depth Stream">
        </div>
        <div class="panel scan-panel">
            <h3>LiDAR Scan (64 bins)</h3>
            <canvas id="scanChart" height="200"></canvas>
        </div>
    </div>

    <div class="status-bar" id="status-bar">
        <div class="status-item"><span class="status-label">Risk:</span><span class="status-value" id="risk-level">--</span></div>
        <div class="status-item"><span class="status-label">Front:</span><span class="status-value" id="front-min">--</span></div>
        <div class="status-item"><span class="status-label">Left:</span><span class="status-value" id="left-min">--</span></div>
        <div class="status-item"><span class="status-label">Right:</span><span class="status-value" id="right-min">--</span></div>
        <div class="status-item"><span class="status-label">Pallet:</span><span class="status-value" id="pallet-detected">--</span></div>
        <div class="status-item"><span class="status-label">FPS:</span><span class="status-value" id="fps">--</span></div>
        <div class="status-item"><span class="status-label">Latency:</span><span class="status-value" id="latency-ms">--</span></div>
        <div class="status-item"><span class="status-label">Shield:</span><span class="status-value shield-active" id="shield-active" style="display:none;color:#f00;">ACTIVE</span></div>
    </div>

    <script>
        // Scan chart
        const ctx = document.getElementById('scanChart').getContext('2d');
        const scanChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 64}, (_, i) => i),
                datasets: [{
                    label: 'Range (m)',
                    data: Array(64).fill(0),
                    borderColor: '#0ff',
                    backgroundColor: 'rgba(0, 255, 255, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { title: { display: true, text: 'Scan Bin', color: '#888' },
                         ticks: { color: '#888' }, grid: { color: '#333' } },
                    y: { title: { display: true, text: 'Distance (m)', color: '#888' },
                         ticks: { color: '#888' }, grid: { color: '#333' },
                         min: 0, max: 5, reverse: true }
                },
                plugins: {
                    legend: { labels: { color: '#888' } },
                    annotation: {
                        annotations: {
                            dangerLine: { type: 'line', yMin: 0.2, yMax: 0.2, borderColor: '#f00', borderWidth: 1, borderDash: [4, 4] },
                            warningLine: { type: 'line', yMin: 0.5, yMax: 0.5, borderColor: '#fc0', borderWidth: 1, borderDash: [4, 4] }
                        }
                    }
                }
            }
        });

        // Poll API every 200ms
        async function pollState() {
            try {
                const resp = await fetch('/api/state');
                const state = await resp.json();

                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').style.color = '#0f0';

                // Risk level styling
                const riskEl = document.getElementById('risk-level');
                riskEl.textContent = state.risk_level.toUpperCase();
                riskEl.className = 'status-value ' + state.risk_level;

                document.getElementById('front-min').textContent = (state.front_min || 0).toFixed(2) + ' m';
                document.getElementById('left-min').textContent = (state.left_min || 0).toFixed(2) + ' m';
                document.getElementById('right-min').textContent = (state.right_min || 0).toFixed(2) + ' m';
                document.getElementById('pallet-detected').textContent = state.pallet_detected ? 'YES' : 'no';
                document.getElementById('fps').textContent = (state.fps || 0).toFixed(1);
                document.getElementById('latency-ms').textContent = (state.latency_ms || 0).toFixed(1) + ' ms';

                const shieldEl = document.getElementById('shield-active');
                if (state.shield_active) {
                    shieldEl.style.display = 'inline';
                } else {
                    shieldEl.style.display = 'none';
                }
            } catch (e) {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').style.color = '#f00';
            }

            // Fetch scan data
            try {
                const scanResp = await fetch('/api/scan');
                const scanData = await scanResp.json();
                if (scanData.scan && scanData.scan.length > 0) {
                    scanChart.data.datasets[0].data = scanData.scan;
                    scanChart.update('none');
                }
            } catch (e) {}

            setTimeout(pollState, 200);
        }

        pollState();
    </script>
</body>
</html>
"""
