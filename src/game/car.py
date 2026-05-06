"""
GripAI — Car
Represents an F1 car: state, physics properties, and drawing.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CarState:
    x: float
    y: float
    angle: float      # radians, 0 = right
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    angular_vel: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    wheel_angle: float = 0.0   # visual front wheel turn
    skid: float = 0.0          # 0-1 tyre slip for effects


class Car:
    """
    Represents a racing car: holds physics state and draws itself on a canvas.
    """

    # Physical dimensions (pixels)
    LENGTH = 28
    WIDTH = 14

    def __init__(
        self,
        start_pos: Tuple[float, float],
        start_angle: float,
        color: Tuple[int, int, int],
        label: str,
    ):
        self.x = float(start_pos[0])
        self.y = float(start_pos[1])
        self.angle = float(start_angle)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.angular_vel = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.wheel_angle = 0.0
        self.skid = 0.0

        self.color = color
        self.label = label

        # Lap / checkpoint state
        self.next_checkpoint = 0
        self.checkpoints_hit = 0

        # Trail for motion effect
        self._trail: list = []
        self._trail_max = 18

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def forward(self) -> Tuple[float, float]:
        return (math.cos(self.angle), math.sin(self.angle))

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, canvas: np.ndarray):
        # Trail
        self._trail.append((int(self.x), int(self.y)))
        if len(self._trail) > self._trail_max:
            self._trail.pop(0)
        self._draw_trail(canvas)

        # Body
        self._draw_body(canvas)

        # Label
        self._draw_label(canvas)

    def _draw_trail(self, canvas: np.ndarray):
        n = len(self._trail)
        for i in range(1, n):
            alpha = i / n
            c = tuple(int(ch * alpha * 0.45) for ch in self.color)
            thickness = max(1, int(alpha * 4))
            cv2.line(canvas, self._trail[i - 1], self._trail[i], c, thickness, cv2.LINE_AA)

    def _draw_body(self, canvas: np.ndarray):
        """Draw rotated rectangle car body with a cockpit bubble and front wing."""
        cx, cy = int(self.x), int(self.y)
        hw = self.WIDTH / 2
        hl = self.LENGTH / 2
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def rot(dx, dy):
            return (
                int(cx + dx * cos_a - dy * sin_a),
                int(cy + dx * sin_a + dy * cos_a),
            )

        # Main body corners
        corners = [rot(-hl, -hw), rot(hl, -hw), rot(hl, hw), rot(-hl, hw)]
        pts = np.array(corners, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], self.color)

        # Cockpit highlight
        cockpit = [rot(-4, -hw + 3), rot(6, -hw + 3), rot(6, hw - 3), rot(-4, hw - 3)]
        cockpit_pts = np.array(cockpit, np.int32).reshape(-1, 1, 2)
        darker = tuple(max(0, ch - 60) for ch in self.color)
        cv2.fillPoly(canvas, [cockpit_pts], darker)

        # Front wing (thin bar ahead)
        wing_l = [rot(hl, -hw - 4), rot(hl + 4, -hw - 4),
                  rot(hl + 4, hw + 4), rot(hl, hw + 4)]
        wing_pts = np.array(wing_l, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [wing_pts], (200, 200, 200))

        # Outline
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Skid spark effect
        if self.skid > 0.3:
            for _ in range(int(self.skid * 6)):
                ox = np.random.randint(-10, 10)
                oy = np.random.randint(-10, 10)
                cv2.circle(canvas, (cx + ox, cy + oy), 1, (80, 200, 255), -1, cv2.LINE_AA)

    def _draw_label(self, canvas: np.ndarray):
        cv2.putText(
            canvas, self.label,
            (int(self.x) - 10, int(self.y) - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
