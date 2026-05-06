"""
GripAI — Track
Defines the racing circuit as a spline-based path with inner/outer boundaries,
checkpoints, and constraint enforcement.
"""

import math
import numpy as np
import cv2
from typing import List, Tuple, Optional


class Track:
    """
    A closed oval/circuit track defined by a centre-line spine.
    Generates road surface, kerbs, start/finish line, and checkpoints.
    """

    ROAD_WIDTH = 90       # pixels, half-width of driveable surface
    KERB_WIDTH = 10
    CHECKPOINT_RADIUS = ROAD_WIDTH + 5

    def __init__(self, canvas_w: int, canvas_h: int):
        self.name = "Circuit Monza Pixel"
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # Generate centre-line points (closed circuit)
        self.centre_line = self._build_circuit(canvas_w, canvas_h)

        # Precompute checkpoints along the spine
        self.checkpoints = self._build_checkpoints(step=12)

        # Start position (first checkpoint, slightly before line)
        sp = self.centre_line[5]
        sp2 = self.centre_line[6]
        self.start_angle = math.atan2(sp2[1] - sp[1], sp2[0] - sp[0])
        self.start_position = (float(sp[0]), float(sp[1]) + 20)
        self.ai_start_position = (float(sp[0]), float(sp[1]) - 20)

        # Pre-render track surface onto a background image
        self._bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        self._render_track(self._bg)

    # ------------------------------------------------------------------
    # Circuit geometry
    # ------------------------------------------------------------------

    def _build_circuit(self, w: int, h: int) -> List[Tuple[int, int]]:
        """
        Returns a list of (x, y) points forming a closed smooth racing circuit.
        Uses a parametric shape: elongated oval with chicanes.
        """
        cx, cy = w // 2, h // 2
        rx, ry = w * 0.38, h * 0.36
        pts = []
        n = 300

        for i in range(n):
            t = 2 * math.pi * i / n
            # Base ellipse
            x = cx + rx * math.cos(t)
            y = cy + ry * math.sin(t)
            # Chicane distortion on the straights
            if 0.15 < (t % (math.pi)) < 0.55:
                x += 40 * math.sin(t * 5) * math.cos(t)
                y += 25 * math.cos(t * 4)
            pts.append((int(x), int(y)))

        return pts

    def _build_checkpoints(self, step: int) -> List[Tuple[int, int]]:
        """Sample every `step`-th point as a checkpoint."""
        return self.centre_line[::step]

    # ------------------------------------------------------------------
    # Track rendering (called once into background)
    # ------------------------------------------------------------------

    def _render_track(self, img: np.ndarray):
        pts = np.array(self.centre_line, np.int32).reshape(-1, 1, 2)

        # Grass base
        img[:] = (20, 60, 20)

        # Outer kerb (red/white)
        cv2.polylines(img, [pts], True, (200, 200, 200), self.ROAD_WIDTH * 2 + self.KERB_WIDTH * 2, cv2.LINE_AA)
        # Alternating kerb colours
        n = len(self.centre_line)
        for i in range(0, n - 1, 8):
            color = (220, 50, 50) if (i // 8) % 2 == 0 else (240, 240, 240)
            seg = np.array(self.centre_line[i:i + 9], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [seg], False, color,
                          self.ROAD_WIDTH * 2 + self.KERB_WIDTH * 2, cv2.LINE_AA)

        # Road surface (dark tarmac)
        cv2.polylines(img, [pts], True, (50, 50, 55), self.ROAD_WIDTH * 2, cv2.LINE_AA)

        # Centre line (dashed white)
        for i in range(0, n - 1, 12):
            if (i // 12) % 2 == 0:
                p1 = self.centre_line[i]
                p2 = self.centre_line[min(i + 6, n - 1)]
                cv2.line(img, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        # Start / finish line (chequered strip)
        self._draw_start_finish(img)

        # Tyre barrier dots (decorative)
        for i in range(0, n, 25):
            self._draw_barriers(img, i)

    def _draw_start_finish(self, img: np.ndarray):
        cx, cy = int(self.centre_line[0][0]), int(self.centre_line[0][1])
        angle = self.start_angle + math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw = self.ROAD_WIDTH
        p1 = (int(cx - cos_a * hw), int(cy - sin_a * hw))
        p2 = (int(cx + cos_a * hw), int(cy + sin_a * hw))
        # White/black squares
        for k in range(-4, 5):
            c = (0, 0, 0) if k % 2 == 0 else (255, 255, 255)
            px = int(cx + cos_a * k * 10)
            py = int(cy + sin_a * k * 10)
            cv2.rectangle(img, (px - 5, py - 5), (px + 5, py + 5), c, -1)

    def _draw_barriers(self, img: np.ndarray, idx: int):
        n = len(self.centre_line)
        pt = self.centre_line[idx % n]
        nxt = self.centre_line[(idx + 1) % n]
        tang = (nxt[0] - pt[0], nxt[1] - pt[1])
        mag = math.hypot(*tang) + 1e-9
        perp = (-tang[1] / mag, tang[0] / mag)
        for side in (-1, 1):
            bx = int(pt[0] + perp[0] * (self.ROAD_WIDTH + self.KERB_WIDTH + 6) * side)
            by = int(pt[1] + perp[1] * (self.ROAD_WIDTH + self.KERB_WIDTH + 6) * side)
            cv2.circle(img, (bx, by), 4, (40, 40, 40), -1)

    # ------------------------------------------------------------------
    # Runtime: draw + constraints
    # ------------------------------------------------------------------

    def draw(self, canvas: np.ndarray):
        """Blit the pre-rendered track background."""
        np.copyto(canvas, self._bg)

    def apply_constraints(self, car):
        """Push car back onto the road if it strays off the track."""
        dist, closest_pt, tang = self._closest_point(car.x, car.y)
        if dist > self.ROAD_WIDTH:
            # Normal direction toward track centre
            perp = np.array([car.x - closest_pt[0], car.y - closest_pt[1]], float)
            perp_mag = np.linalg.norm(perp)
            if perp_mag > 1e-9:
                perp /= perp_mag
            overshoot = dist - self.ROAD_WIDTH
            car.x -= perp[0] * overshoot
            car.y -= perp[1] * overshoot
            # Dampen velocity component toward wall
            vel = np.array([car.vx, car.vy])
            dot = np.dot(vel, perp)
            if dot > 0:
                car.vx -= perp[0] * dot * 0.7
                car.vy -= perp[1] * dot * 0.7

    def check_checkpoint(self, car) -> Optional[int]:
        """Return checkpoint index if car crosses it, else None."""
        for i, cp in enumerate(self.checkpoints):
            dx = car.x - cp[0]
            dy = car.y - cp[1]
            if math.hypot(dx, dy) < self.CHECKPOINT_RADIUS:
                return i
        return None

    def get_path_ahead(self, x: float, y: float, n: int = 8) -> List[Tuple[int, int]]:
        """Returns next n centre-line points ahead of (x, y). Used by AI."""
        _, _, closest_idx = self._closest_point_idx(x, y)
        pts = []
        total = len(self.centre_line)
        for i in range(1, n + 1):
            pts.append(self.centre_line[(closest_idx + i * 3) % total])
        return pts

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _closest_point(self, x: float, y: float):
        _, closest, idx = self._closest_point_idx(x, y)
        n = len(self.centre_line)
        nxt = self.centre_line[(idx + 1) % n]
        tang = (nxt[0] - closest[0], nxt[1] - closest[1])
        return math.hypot(x - closest[0], y - closest[1]), closest, tang

    def _closest_point_idx(self, x: float, y: float):
        best_d = float("inf")
        best_pt = self.centre_line[0]
        best_i = 0
        for i, pt in enumerate(self.centre_line):
            d = math.hypot(x - pt[0], y - pt[1])
            if d < best_d:
                best_d = d
                best_pt = pt
                best_i = i
        return best_d, best_pt, best_i
