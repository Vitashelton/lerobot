"""
Pure-numpy depth renderer (rasterizer) for synthetic scenes.

Uses a Z-buffer approach with painter's algorithm:
  1. Sort faces / surfaces by distance from camera.
  2. Rasterize each face into pixel space.
  3. Per-pixel Z-buffer test -- only write if candidate depth is closer.

Camera model: pinhole with intrinsics (fx, fy, cx, cy).
World convention: X=right, Y=up, Z=forward (depth).
Camera looks along +Z; pitch rotates around the X axis (nose-down = positive).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix around world X axis (pitch)."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,   -s],
                     [0.0, s,    c]], dtype=np.float64)


class SyntheticDepthRenderer:
    """Z-buffer depth rasterizer for synthetic 3D scenes.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    fx, fy : float
        Focal lengths (pixels).  Typically fx == fy for square pixels.
    cx, cy : float
        Principal point (pixels), typically near (width/2, height/2).
    """

    def __init__(self, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    # ------------------------------------------------------------------
    #  Low-level helpers
    # ------------------------------------------------------------------

    def _world_to_camera(self, point_world: np.ndarray,
                         camera_pos: np.ndarray,
                         pitch_rad: float) -> np.ndarray:
        """Transform a 3-D point from world frame to camera frame.

        Camera frame: X_c = right, Y_c = down, Z_c = forward.
        """
        R = _rotation_x(-pitch_rad)  # undo world pitch → camera pitch
        p = point_world - camera_pos
        return R @ p  # (3,) or (N,3)

    def _project(self, point_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3-D camera points to pixel coordinates.

        Returns
        -------
        u, v : ndarray
            Pixel coordinates.  Values may lie outside [0, W/H).
        depth : ndarray
            Z component in camera frame (metres).
        """
        X = point_camera[..., 0]
        Y = point_camera[..., 1]
        Z = point_camera[..., 2]
        # guard against division by zero / negative Z
        Z_safe = np.where(np.abs(Z) < 1e-6, np.sign(Z + 1e-12) * 1e-6, Z)
        u = self.fx * X / Z_safe + self.cx
        v = self.fy * Y / Z_safe + self.cy
        return u, v, Z

    # ------------------------------------------------------------------
    #  Public rendering entry points
    # ------------------------------------------------------------------

    def render_box(self, center_3d: Tuple[float, float, float],
                   size_3d: Tuple[float, float, float],
                   rotation_yaw: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render a 3-D cuboid into pixel space.

        Parameters
        ----------
        center_3d : (cx, cy, cz)  world frame, metres.
        size_3d   : (wx, wy, wz)  full extents, metres.
        rotation_yaw : float  yaw about world Y axis (radians).

        Returns
        -------
        pixel_coords : ndarray (N_faces, 4, 2)  image coords for each face corner.
        depth_values : ndarray (N_faces, 4)    camera depth at each corner.
        face_mask    : ndarray (N_faces,) bool  True = face visible (front-facing).
        """
        cx, cy, cz = center_3d
        sx, sy, sz = size_3d
        hx, hy, hz = sx / 2, sy / 2, sz / 2

        # 8 corners in world, centred at origin
        corners_local = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ], dtype=np.float64)

        # Yaw rotation
        c = np.cos(rotation_yaw)
        s = np.sin(rotation_yaw)
        Ry = np.array([[c, 0, s],
                       [0, 1, 0],
                       [-s, 0, c]], dtype=np.float64)
        corners_world = (Ry @ corners_local.T).T + np.array([cx, cy, cz])

        # 6 faces (indices into corners array)
        # Order: -Z, +Z, -X, +X, -Y, +Y  (back, front, left, right, bottom, top)
        face_indices = np.array([
            [0, 1, 2, 3],  # back  (-Z)
            [4, 5, 6, 7],  # front (+Z)
            [0, 3, 7, 4],  # left  (-X)
            [1, 2, 6, 5],  # right (+X)
            [0, 1, 5, 4],  # bottom (-Y)
            [3, 2, 6, 7],  # top    (+Y)
        ], dtype=np.intp)

        N_faces = 6
        pixel_coords = np.zeros((N_faces, 4, 2), dtype=np.float32)
        depth_values = np.zeros((N_faces, 4), dtype=np.float32)
        face_mask = np.ones(N_faces, dtype=bool)

        for i, idx in enumerate(face_indices):
            verts_world = corners_world[idx]       # (4, 3)
            u, v, Z = self._project(verts_world)   # camera must already be applied
            # ^ only valid after we transform to camera.
            # For now we store raw values to be processed later.
            # We defer projection to the scene-level call where camera pose is known.
            pixel_coords[i] = np.column_stack([u, v])
            depth_values[i] = Z

        return pixel_coords, depth_values, face_mask

    # ------------------------------------------------------------------
    #  Rasterize a single convex quad face with per-pixel depth
    # ------------------------------------------------------------------

    def _rasterize_quad(self, corners_3d: np.ndarray,
                        depth_map: np.ndarray,
                        label_map: np.ndarray,
                        rgb: np.ndarray,
                        color: Tuple[int, int, int],
                        label_id: int,
                        z_buffer: np.ndarray,
                        ) -> None:
        """Rasterize a 3-D quad face into the accumulation buffers.

        Only pixels whose projected depth is *closer* than z_buffer are written.

        Parameters
        ----------
        corners_3d : ndarray (4, 3)  vertices in **camera** frame.
        depth_map  : ndarray (H, W) float32   (output) clean depth.
        label_map  : ndarray (H, W) int32     (output) semantic labels.
        rgb        : ndarray (H, W, 3) uint8  (output) colour.
        color      : (R, G, B) uint8 tuple.
        label_id   : int  class id to write into label_map.
        z_buffer   : ndarray (H, W) float32   (in/out) Z-buffer.

        Notes
        -----
        Depth is computed per-pixel from the plane equation of the face.
        A pixel inside the quad is written only if its depth < z_buffer[pixel].
        """
        H, W = depth_map.shape
        u = np.zeros(4, dtype=np.float32)
        v = np.zeros(4, dtype=np.float32)
        Zc = np.zeros(4, dtype=np.float32)

        for k in range(4):
            X, Y, Z = corners_3d[k]
            Z_s = max(abs(Z), 1e-6)
            u[k] = self.fx * X / Z_s + self.cx
            v[k] = self.fy * Y / Z_s + self.cy
            Zc[k] = Z

        # Axis-aligned bounding box in pixel space
        u_min = int(np.floor(np.clip(np.min(u), 0, W - 1)))
        u_max = int(np.ceil(np.clip(np.max(u), 0, W - 1)))
        v_min = int(np.floor(np.clip(np.min(v), 0, H - 1)))
        v_max = int(np.ceil(np.clip(np.max(v), 0, H - 1)))

        if u_max < u_min or v_max < v_min:
            return

        # Plane equation: ax + by + cz = d  in camera space.
        # Build from first three vertices.
        p0, p1, p2 = corners_3d[0], corners_3d[1], corners_3d[2]
        normal = np.cross(p1 - p0, p2 - p0)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            return
        normal /= n_len
        d = np.dot(normal, p0)

        # Edge vectors for inside-outside test (2-D)
        edges = np.zeros((4, 2), dtype=np.float32)
        for k in range(4):
            nxt = (k + 1) % 4
            edges[k, 0] = u[nxt] - u[k]   # du
            edges[k, 1] = v[nxt] - v[k]   # dv

        # Raster loop
        uu = np.arange(u_min, u_max + 1, dtype=np.float32)
        vv = np.arange(v_min, v_max + 1, dtype=np.float32)
        ug, vg = np.meshgrid(uu, vv)  # (rows, cols)

        # Inside test: check each edge
        inside = np.ones_like(ug, dtype=bool)
        for k in range(4):
            # vector from vertex k to pixel
            dp_u = ug - u[k]
            dp_v = vg - v[k]
            cross = edges[k, 0] * dp_v - edges[k, 1] * dp_u
            inside &= (cross >= -0.5)  # small tolerance

        if not np.any(inside):
            return

        # Compute per-pixel depth from plane equation
        # For pixel (u,v): ray direction in camera space =
        #   ((u-cx)/fx, (v-cy)/fy, 1.0)
        ray_x = (ug - self.cx) / self.fx
        ray_y = (vg - self.cy) / self.fy
        # dot(n, ray_dir) * Z = d  →  Z = d / dot(n, ray_dir)
        denom = normal[0] * ray_x + normal[1] * ray_y + normal[2]
        denom_safe = np.where(np.abs(denom) < 1e-9, np.sign(denom + 1e-12) * 1e-9, denom)
        pixel_depth = d / denom_safe

        # Only write pixels that are inside AND closer than current Z-buffer
        # AND have positive depth
        write_mask = inside & (pixel_depth > 0.0) & (pixel_depth < z_buffer[v_min:v_max + 1, u_min:u_max + 1])

        # Extract slice views for the bounding box
        depth_slice = depth_map[v_min:v_max + 1, u_min:u_max + 1]
        label_slice = label_map[v_min:v_max + 1, u_min:u_max + 1]
        zbuf_slice = z_buffer[v_min:v_max + 1, u_min:u_max + 1]

        depth_slice[write_mask] = pixel_depth[write_mask]
        label_slice[write_mask] = label_id
        zbuf_slice[write_mask] = pixel_depth[write_mask]

        # Colour
        rgb_slice = rgb[v_min:v_max + 1, u_min:u_max + 1]
        for c in range(3):
            rgb_slice[write_mask, c] = color[c]

    # ------------------------------------------------------------------
    #  Planar surface (floor / wall)
    # ------------------------------------------------------------------

    def render_planar_surface(self,
                              plane_origin: Tuple[float, float, float],
                              plane_normal: Tuple[float, float, float],
                              plane_width: float,
                              plane_height: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vertex data for a rectangular planar surface.

        Parameters
        ----------
        plane_origin : (ox, oy, oz)  centre of the rectangle, world frame.
        plane_normal : (nx, ny, nz)  unit normal (world frame).
        plane_width  : float  extent along the local U axis.
        plane_height : float  extent along the local V axis.

        Returns
        -------
        corners_3d : ndarray (4, 3)  world-frame corners.
        normal_3d  : ndarray (3,)    world-frame normal.
        valid      : bool            always True for this implementation.
        """
        ox, oy, oz = plane_origin
        nx, ny, nz = plane_normal
        n = np.array([nx, ny, nz], dtype=np.float64)
        n_norm = n / np.linalg.norm(n)

        # Build a local basis {u, v} orthogonal to n
        # Pick an arbitrary reference vector not parallel to n
        if abs(n_norm[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u_axis = np.cross(n_norm, ref)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(n_norm, u_axis)
        v_axis /= np.linalg.norm(v_axis)

        hw = plane_width / 2
        hh = plane_height / 2
        origin = np.array([ox, oy, oz], dtype=np.float64)
        corners = np.array([
            origin - hw * u_axis - hh * v_axis,
            origin + hw * u_axis - hh * v_axis,
            origin + hw * u_axis + hh * v_axis,
            origin - hw * u_axis + hh * v_axis,
        ], dtype=np.float64)

        return corners, n_norm, True

    # ------------------------------------------------------------------
    #  Cylinder (person / chair approximation)
    # ------------------------------------------------------------------

    def render_cylinder(self,
                        center_3d: Tuple[float, float, float],
                        radius: float,
                        height: float,
                        num_faces: int = 16
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate a vertical cylinder with N planar faces.

        Parameters
        ----------
        center_3d : (cx, cy, cz)   world frame.
        radius    : float          cylinder radius (metres).
        height    : float          cylinder height (metres).
        num_faces : int            number of planar faces around the circle.

        Returns
        -------
        all_corners : ndarray (num_faces * 4, 3)  world-frame corners, grouped by face.
        all_normals : ndarray (num_faces, 3)      outward normals per face.
        valid       : ndarray (num_faces,) bool   True for each face.
        """
        cx, cy, cz = center_3d
        angles = np.linspace(0, 2 * np.pi, num_faces, endpoint=False)
        bottom_y = cy - height / 2
        top_y = cy + height / 2

        all_corners = np.zeros((num_faces * 4, 3), dtype=np.float64)
        all_normals = np.zeros((num_faces, 3), dtype=np.float64)

        for i in range(num_faces):
            a0 = angles[i]
            a1 = angles[(i + 1) % num_faces]
            x0 = cx + radius * np.cos(a0)
            z0 = cz + radius * np.sin(a0)
            x1 = cx + radius * np.cos(a1)
            z1 = cz + radius * np.sin(a1)

            # Outward normal (average direction of the two radians)
            mid_angle = (a0 + a1) / 2
            nx = np.cos(mid_angle)
            nz = np.sin(mid_angle)

            all_corners[i * 4 + 0] = [x0, bottom_y, z0]
            all_corners[i * 4 + 1] = [x1, bottom_y, z1]
            all_corners[i * 4 + 2] = [x1, top_y,    z1]
            all_corners[i * 4 + 3] = [x0, top_y,    z0]
            all_normals[i] = [nx, 0.0, nz]

        valid = np.ones(num_faces, dtype=bool)
        return all_corners, all_normals, valid

    # ------------------------------------------------------------------
    #  Full-scene rendering (called by SceneGenerator)
    # ------------------------------------------------------------------

    def compose_scene(self,
                      objects: list,
                      camera_pos: np.ndarray,
                      pitch_rad: float,
                      z_far: float = 30.0,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render all objects into depth, label, and RGB buffers.

        Parameters
        ----------
        objects : list of dict
            Each dict has keys: 'type' ('floor','wall','box','cylinder'),
            plus geometry keys (varies by type).  Objects are rendered
            in the order given; the caller should back-to-front sort.

        camera_pos : ndarray (3,)  world position.
        pitch_rad  : float         pitch angle (nose-down positive).

        Returns
        -------
        depth_map : ndarray (H, W) float32   metres.
        label_map : ndarray (H, W) int32     class IDs.
        rgb       : ndarray (H, W, 3) uint8
        """
        H, W = self.height, self.width
        depth_map = np.full((H, W), z_far, dtype=np.float32)
        label_map = np.zeros((H, W), dtype=np.int32)
        rgb = np.full((H, W, 3), 128, dtype=np.uint8)  # grey background
        z_buffer = np.full((H, W), z_far, dtype=np.float32)

        R = _rotation_x(-pitch_rad)

        for obj in objects:
            obj_type = obj["type"]
            color = obj.get("color", (128, 128, 128))
            label_id = obj.get("label_id", 0)

            if obj_type == "floor":
                corners_w, normal_w, _ = self.render_planar_surface(
                    obj["origin"], obj["normal"], obj["width"], obj["height"])
                corners_cam = (R @ (corners_w - camera_pos).T).T
                self._rasterize_quad(corners_cam, depth_map, label_map,
                                     rgb, color, label_id, z_buffer)

            elif obj_type == "wall":
                corners_w, normal_w, _ = self.render_planar_surface(
                    obj["origin"], obj["normal"], obj["width"], obj["height"])
                corners_cam = (R @ (corners_w - camera_pos).T).T
                # Check front-facing
                n_cam = R @ normal_w
                cam_dir = np.array([0.0, 0.0, 1.0])
                if np.dot(n_cam, cam_dir) >= 0:
                    continue  # back-facing
                self._rasterize_quad(corners_cam, depth_map, label_map,
                                     rgb, color, label_id, z_buffer)

            elif obj_type == "box":
                cx_ = obj["center"][0]
                cy_ = obj["center"][1]
                cz_ = obj["center"][2]
                sx_ = obj["size"][0]
                sy_ = obj["size"][1]
                sz_ = obj["size"][2]
                yaw = obj.get("yaw", 0.0)

                hx, hy, hz = sx_ / 2, sy_ / 2, sz_ / 2
                corners_local = np.array([
                    [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                    [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
                ], dtype=np.float64)

                c_yaw = np.cos(yaw); s_yaw = np.sin(yaw)
                Ry = np.array([[c_yaw, 0, s_yaw], [0, 1, 0], [-s_yaw, 0, c_yaw]], dtype=np.float64)
                corners_w = (Ry @ corners_local.T).T + np.array([cx_, cy_, cz_])

                face_idx = np.array([
                    [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4],
                    [1, 2, 6, 5], [0, 1, 5, 4], [3, 2, 6, 7],
                ], dtype=np.intp)

                for fi in face_idx:
                    verts_w = corners_w[fi]
                    verts_cam = (R @ (verts_w - camera_pos).T).T
                    # front-face culling: compute face normal in camera space
                    e1 = verts_cam[1] - verts_cam[0]
                    e2 = verts_cam[2] - verts_cam[0]
                    n_cam = np.cross(e1, e2)
                    # The face is visible if the normal points toward the camera
                    # which sits at origin in camera space; i.e. n · mean_vert < 0
                    mean_vert = verts_cam.mean(axis=0)
                    if np.dot(n_cam, mean_vert) >= 0:
                        continue
                    self._rasterize_quad(verts_cam, depth_map, label_map,
                                         rgb, color, label_id, z_buffer)

            elif obj_type == "cylinder":
                cx_ = obj["center"][0]
                cy_ = obj["center"][1]
                cz_ = obj["center"][2]
                r = obj["radius"]
                h = obj["height"]
                n_f = obj.get("num_faces", 16)

                all_c, all_n, _ = self.render_cylinder(
                    (cx_, cy_, cz_), r, h, n_f)

                for i in range(n_f):
                    verts_w = all_c[i * 4:i * 4 + 4]
                    n_w = all_n[i]
                    verts_cam = (R @ (verts_w - camera_pos).T).T
                    n_cam = R @ n_w
                    mean_vert = verts_cam.mean(axis=0)
                    if np.dot(n_cam, mean_vert) >= 0:
                        continue
                    self._rasterize_quad(verts_cam, depth_map, label_map,
                                         rgb, color, label_id, z_buffer)

        return depth_map, label_map, rgb
