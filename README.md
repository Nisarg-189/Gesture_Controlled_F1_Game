<div align="center">

```
 ██████╗ ███████╗███████╗████████╗██╗   ██╗██████╗ ███████╗
██╔════╝ ██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
██║  ███╗█████╗  ███████╗   ██║   ██║   ██║██████╔╝█████╗
██║   ██║██╔══╝  ╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝
╚██████╔╝███████╗███████║   ██║   ╚██████╔╝██║  ██║███████╗
 ╚═════╝ ╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝

 ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  ██████╗ ██╗     ██╗     ███████╗██████╗
██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██║     ██║     ██╔════╝██╔══██╗
██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║   ██║██║     ██║     █████╗  ██║  ██║
██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║██║     ██║     ██╔══╝  ██║  ██║
╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗███████╗███████╗██████╔╝
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═════╝

██████╗  █████╗  ██████╗██╗███╗   ██╗ ██████╗ 
██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝ 
██████╔╝███████║██║     ██║██╔██╗ ██║██║  ███╗
██╔══██╗██╔══██║██║     ██║██║╚██╗██║██║   ██║
██║  ██║██║  ██║╚██████╗██║██║ ╚████║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 

 ██████╗  █████╗ ███╗   ███╗███████╗
██╔════╝ ██╔══██╗████╗ ████║██╔════╝
██║  ███╗███████║██╔████╔██║█████╗  
██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  
╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
```

**Gesture-Controlled F1 Racing · AI That Learns Your Style**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange?style=flat-square)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

*Wave your hand. Race an AI. Watch it learn you.*

</div>

---

## 🎮 What is GripAI?

**GripAI** is a real-time F1 mini-game where you control a racing car using **hand gestures** captured via your webcam — no controller, no keyboard (unless you want one).

The twist: your opponent isn't just following a script. The **AI observes your driving style** — your aggression, your braking points, how fast you take corners — and **adapts its behaviour** to mirror, challenge, and eventually outmanoeuvre you.

The longer the race, the more dangerous the AI gets.

```
✊ Fist          →  BRAKE
✋ Open Palm     →  ACCELERATE  (openness = throttle intensity)
👈 Tilt Left     →  STEER LEFT
👉 Tilt Right    →  STEER RIGHT
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🖐️ **Gesture Control** | 21-landmark MediaPipe hand model → real-time throttle / brake / steer |
| 🤖 **Adaptive AI** | Pure-pursuit path follower that builds a statistical profile of your driving style |
| 🏎️ **F1 Physics** | Bicycle model with tyre temperature, lateral slip, downforce, and drag |
| 🏁 **Full Circuit** | Parametric spline track with kerbs, start/finish line, and animated trail |
| 📊 **Live HUD** | Speed, lap times, best lap, AI adaptation meter, camera PiP |
| ⌨️ **Keyboard Fallback** | WASD / arrow keys when no webcam is available |
| 🔧 **Configurable** | Difficulty modes: `easy`, `medium`, `hard`, `adaptive` |

---

## 🏗️ Project Structure

```
GripAI/
│
├── main.py                     # Entry point — launch everything from here
│
├── src/
│   ├── game/
│   │   ├── engine.py           # Core game loop, state machine, rendering
│   │   ├── car.py              # Car state, drawing (body, trail, sparks)
│   │   ├── physics.py          # F1 tyre/bicycle physics engine
│   │   ├── track.py            # Spline circuit, checkpoints, constraints
│   │   └── hud.py              # All HUD overlays (racing, menu, results)
│   │
│   ├── vision/
│   │   └── gesture_controller.py   # MediaPipe → throttle/brake/steer
│   │
│   ├── ai/
│   │   └── opponent.py         # Adaptive AI: pure pursuit + style mirror
│   │
│   └── utils/
│       └── logger.py           # Shared logger factory
│
├── tests/
│   └── test_core.py            # Unit tests: physics, AI profile, adaptation
│
├── assets/                     # Sounds and sprites (extend here)
├── docs/                       # Architecture diagrams, notes
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/GripAI.git
cd GripAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required. Tested on Ubuntu 22.04, macOS 13, Windows 11.

### 3. Run the game

```bash
# Default: gesture control, adaptive AI, 60 FPS
python main.py

# Keyboard only (no webcam needed)
python main.py --no-gesture

# Hard AI, debug overlays
python main.py --ai-difficulty hard --debug

# Custom resolution and camera index
python main.py --width 1920 --height 1080 --camera 1
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## 🖐️ Gesture Guide

```
┌────────────────────────────────────────────────────┐
│                                                    │
│   ✊  FIST         →  Full Brake                  │
│                                                    │
│   ✋  OPEN PALM    →  Throttle                    │
│       (wider spread = more throttle)               │
│                                                    │
│   ↙  TILT LEFT    →  Steer Left                  │
│   ↘  TILT RIGHT   →  Steer Right                 │
│                                                    │
│   No hand visible  →  Coast                       │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Tips:**
- Keep your hand **30–60 cm** from the camera
- Good **lighting** dramatically improves tracking
- Use a **plain background** if detection is inconsistent
- The camera PiP in the bottom-right shows what the model sees

---

## 🤖 How the AI Adaptation Works

The AI opponent runs in two layers:

**Layer 1 — Path Follower**
A pure-pursuit controller with PD steering looks ahead along the circuit spline and steers toward a lookahead point scaled by speed. Speed is managed via curvature estimation (slow for corners, push on straights).

**Layer 2 — Style Mirror**
Every game tick, the AI records your telemetry into a rolling window:
- **Aggression score** — average throttle minus scaled brake usage
- **Corner speed** — average speed when you're steering hard
- **Braking sharpness** — how hard you brake on average

As the `adaptation_level` (shown on HUD) grows from 0% → 100%, the AI's own target speed, braking distance, and corner slowdown factor **blend toward your profile**. If you drive flat-out and late-brake, the AI learns to do the same. If you're smooth and conservative, so will it be.

```
Tick 0:      AI uses default base profile
Tick 600:    10% adapted — AI has seen ~10s of your style
Tick 3600:   Full adaptation — AI mirrors your aggression, corner speeds, braking
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--width` | `1280` | Game window width |
| `--height` | `720` | Game window height |
| `--camera` | `0` | Webcam device index |
| `--fps` | `60` | Target frame rate |
| `--ai-difficulty` | `adaptive` | `easy / medium / hard / adaptive` |
| `--no-gesture` | `False` | Disable webcam, use keyboard |
| `--debug` | `False` | Show landmark overlays and debug text |

---

## 🛣️ Roadmap

- [ ] **Multiplayer** — LAN gesture race via sockets
- [ ] **Custom tracks** — JSON-defined spline editor
- [ ] **More gestures** — two-hand DRS activation, peace sign = pit
- [ ] **ML model** — replace rule-based gesture classification with a trained classifier
- [ ] **Sound** — engine rev, tyre squeal, crowd roar via pygame.mixer
- [ ] **Telemetry dashboard** — post-race matplotlib plots of style evolution
- [ ] **Mobile camera** — DroidCam / OBS virtual camera support

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

```bash
# Run tests before submitting
python -m pytest tests/ -v --tb=short
```

---

## 📄 License

MIT — do whatever you want, just don't blame us when you're late for school because you've been racing all night.

---

<div align="center">

Built with ☕ + 🏎️ + way too much interest in tyre physics.

**Gesture Controlled F1 Game** — *The AI grips your style.*

</div>
