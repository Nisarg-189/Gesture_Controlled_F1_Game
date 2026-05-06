"""
GripAI — Gesture Controller
Uses MediaPipe Hands to translate hand pose into throttle / brake / steer signals.

Gesture vocabulary:
  • Open palm (5 fingers spread)  → Throttle (speed = openness)
  • Closed fist                   → Brake
  • Hand tilt left/right          → Steer
  • No hand detected              → Coast (all inputs 0)
"""

import math
import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class GestureInput:
    throttle: float = 0.0   # 0..1
    brake: float = 0.0      # 0..1
    steer: float = 0.0      # -1 (left) .. 1 (right)
    gesture_label: str = "none"
    confidence: float = 0.0


class GestureController:
    """
    Captures webcam frames, runs MediaPipe hand landmark detection,
    and maps pose features to driving inputs.
    """

    # Smoothing
    SMOOTH_ALPHA = 0.35    # EMA weight for input smoothing

    # Steer sensitivity: tilt angle (radians) that maps to full steer
    STEER_MAX_TILT = math.radians(35)

    def __init__(self, camera_index: int = 0, debug: bool = False):
        self.debug = debug
        self.last_frame: Optional[np.ndarray] = None
        self._input = GestureInput()
        self._cap = None
        self._hands = None

        if not MEDIAPIPE_AVAILABLE:
            print("⚠️  mediapipe not installed. Run: pip install mediapipe")
            return

        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_input(self) -> GestureInput:
        """
        Grabs a webcam frame, runs hand detection, and returns a GestureInput.
        Call once per game tick.
        """
        if not self.is_available():
            return GestureInput()

        ret, frame = self._cap.read()
        if not ret:
            return self._input

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        new_input = GestureInput()

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]

            # Draw landmarks on camera PiP
            if self.debug:
                self._mp_draw.draw_landmarks(
                    frame, landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

            new_input = self._landmarks_to_input(landmarks, frame.shape)
        else:
            new_input.gesture_label = "no_hand"

        # Smooth inputs with EMA
        a = self.SMOOTH_ALPHA
        self._input.throttle = a * new_input.throttle + (1 - a) * self._input.throttle
        self._input.brake = a * new_input.brake + (1 - a) * self._input.brake
        self._input.steer = a * new_input.steer + (1 - a) * self._input.steer
        self._input.gesture_label = new_input.gesture_label
        self._input.confidence = new_input.confidence

        # Annotate camera frame with HUD
        self._annotate_frame(frame, self._input)
        self.last_frame = frame

        return self._input

    # ------------------------------------------------------------------
    # Landmark → input mapping
    # ------------------------------------------------------------------

    def _landmarks_to_input(self, landmarks, frame_shape) -> GestureInput:
        h, w = frame_shape[:2]
        lm = landmarks.landmark

        # Key landmark indices (MediaPipe 21-point model)
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13
        PINKY_MCP = 17

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        wrist = pt(WRIST)
        index_mcp = pt(INDEX_MCP)
        middle_mcp = pt(MIDDLE_MCP)

        finger_tips = [pt(INDEX_TIP), pt(MIDDLE_TIP), pt(RING_TIP), pt(PINKY_TIP)]
        finger_bases = [pt(INDEX_MCP), pt(MIDDLE_MCP), pt(RING_MCP), pt(PINKY_MCP)]

        # --- Finger extension ratio (tip distance vs base distance from wrist) ---
        extensions = []
        for tip, base in zip(finger_tips, finger_bases):
            base_d = np.linalg.norm(wrist - base) + 1e-9
            tip_d = np.linalg.norm(wrist - tip)
            extensions.append(min(1.0, tip_d / (base_d * 1.6)))

        openness = sum(extensions) / len(extensions)   # 0 = fist, 1 = open

        # --- Gesture classification ---
        is_fist = openness < 0.35
        is_open = openness > 0.65

        # --- Steering: tilt of palm normal ---
        # Vector from wrist to middle MCP base
        palm_vec = middle_mcp - wrist
        tilt_angle = math.atan2(palm_vec[0], -palm_vec[1])   # 0 = upright
        steer = max(-1.0, min(1.0, tilt_angle / self.STEER_MAX_TILT))

        # --- Throttle / Brake ---
        if is_fist:
            throttle = 0.0
            brake = 1.0
            label = "brake"
        elif is_open:
            throttle = 0.5 + openness * 0.5   # partial to full throttle
            brake = 0.0
            label = "throttle"
        else:
            # Coasting
            throttle = openness * 0.4
            brake = 0.0
            label = "coast"

        # Confidence from detection certainty (approximate via openness stability)
        confidence = abs(openness - 0.5) * 2.0

        return GestureInput(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
            gesture_label=label,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Camera frame annotation
    # ------------------------------------------------------------------

    def _annotate_frame(self, frame: np.ndarray, inp: GestureInput):
        h, w = frame.shape[:2]
        ACCENT = (0, 200, 120)
        # Gesture label
        cv2.putText(frame, inp.gesture_label.upper(), (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT, 2, cv2.LINE_AA)

        # Mini bars for throttle / brake / steer
        bar_labels = [("T", inp.throttle, (0, 200, 80)),
                      ("B", inp.brake, (0, 60, 220)),
                      ("S", (inp.steer + 1) / 2, (200, 120, 0))]
        for i, (lbl, val, col) in enumerate(bar_labels):
            bx, by = 8, h - 55 + i * 16
            cv2.rectangle(frame, (bx, by), (bx + 80, by + 10), (30, 30, 30), -1)
            fill = int(80 * val)
            cv2.rectangle(frame, (bx, by), (bx + fill, by + 10), col, -1)
            cv2.putText(frame, lbl, (bx + 84, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    def release(self):
        if self._cap:
            self._cap.release()
        if self._hands:
            self._hands.close()
