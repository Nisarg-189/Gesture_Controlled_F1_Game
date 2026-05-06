"""
GripAI — Physics Engine
Simplified but expressive F1 car physics:
  - Longitudinal thrust / braking
  - Lateral tyre slip model
  - Drag, downforce, weight transfer
"""

import math
from src.game.car import Car


class PhysicsEngine:
    """
    Updates a Car's velocity and position each tick using a simplified
    tyre-friction / rigid-body model inspired by real F1 dynamics.
    """

    # Engine / powertrain
    MAX_THROTTLE_FORCE = 6500.0   # N equivalent
    MAX_BRAKE_FORCE = 12000.0
    DRAG_COEFF = 0.018
    DOWNFORCE_COEFF = 0.012       # extra grip at speed

    # Tyres
    MAX_LATERAL_ACCEL = 28.0      # m/s²  (F1 pulls ~5g lateral)
    TYRE_GRIP_BASE = 0.92
    TYRE_WARMUP_RATE = 0.004
    TYRE_COOLDOWN_RATE = 0.002

    # Steering
    MAX_STEER_ANGLE = math.radians(28)   # front wheel lock
    STEER_SPEED = 4.0                    # rad/s turn rate of steering wheel
    WHEELBASE = 22.0                     # pixels (car length * 0.8)

    # Mass (pixel-space)
    MASS = 1.0                           # normalised — forces already scaled

    # Speed limit (pixels/s)
    MAX_SPEED = 420.0

    def __init__(self):
        self._tyre_temp: dict = {}   # car id → 0..1 warmth

    def update(self, car: Car, throttle: float, brake: float, steer: float, dt: float):
        """Apply one physics tick to car."""
        car_id = id(car)
        temp = self._tyre_temp.get(car_id, 0.3)

        # Smooth steering wheel
        target_wheel = steer * self.MAX_STEER_ANGLE
        car.wheel_angle += (target_wheel - car.wheel_angle) * min(1.0, self.STEER_SPEED * dt)

        # Current speed along forward axis
        fx = math.cos(car.angle)
        fy = math.sin(car.angle)
        forward_speed = car.vx * fx + car.vy * fy

        # Tyre temperature (warms with slip, cools without)
        lateral_vx = car.vx - forward_speed * fx
        lateral_vy = car.vy - forward_speed * fy
        lateral_speed = math.hypot(lateral_vx, lateral_vy)
        slip_ratio = min(1.0, lateral_speed / max(1.0, abs(forward_speed) + 1.0))
        if slip_ratio > 0.1:
            temp = min(1.0, temp + self.TYRE_WARMUP_RATE)
        else:
            temp = max(0.0, temp - self.TYRE_COOLDOWN_RATE)
        self._tyre_temp[car_id] = temp

        grip = self.TYRE_GRIP_BASE + temp * (1.0 - self.TYRE_GRIP_BASE)
        grip += self.DOWNFORCE_COEFF * forward_speed * 0.01

        # Longitudinal forces
        drive_force = throttle * self.MAX_THROTTLE_FORCE * grip
        brake_force = brake * self.MAX_BRAKE_FORCE * grip

        # Net acceleration along heading
        accel = (drive_force - brake_force) / (self.MASS * 1000.0)
        car.vx += fx * accel * dt
        car.vy += fy * accel * dt

        # Drag (proportional to speed²)
        speed = math.hypot(car.vx, car.vy)
        drag = self.DRAG_COEFF * speed * speed
        if speed > 1e-3:
            car.vx -= (car.vx / speed) * drag * dt
            car.vy -= (car.vy / speed) * drag * dt

        # Lateral friction (prevents infinite sliding)
        speed = math.hypot(car.vx, car.vy)
        lateral_limit = self.MAX_LATERAL_ACCEL * grip * dt
        lat_vx = car.vx - (car.vx * fx + car.vy * fy) * fx
        lat_vy = car.vy - (car.vx * fx + car.vy * fy) * fy
        lat_mag = math.hypot(lat_vx, lat_vy)
        if lat_mag > lateral_limit:
            scale = lateral_limit / lat_mag
            car.vx -= lat_vx * (1.0 - scale)
            car.vy -= lat_vy * (1.0 - scale)

        # Skid value for visual effects
        car.skid = min(1.0, lat_mag / (lateral_limit + 1e-3))

        # Bicycle model steering (angular velocity from Ackermann geometry)
        speed = math.hypot(car.vx, car.vy)
        if speed > 2.0:
            turn_radius = self.WHEELBASE / math.tan(car.wheel_angle + 1e-9)
            angular_vel = speed / turn_radius
        else:
            angular_vel = 0.0

        car.angle += angular_vel * dt
        car.angle = car.angle % (2 * math.pi)

        # Speed cap
        speed = math.hypot(car.vx, car.vy)
        if speed > self.MAX_SPEED:
            car.vx = car.vx / speed * self.MAX_SPEED
            car.vy = car.vy / speed * self.MAX_SPEED

        car.speed = math.hypot(car.vx, car.vy)

        # Position integration
        car.x += car.vx * dt
        car.y += car.vy * dt

        # Store for HUD
        car.throttle = throttle
        car.brake = brake
        car.steer = steer
