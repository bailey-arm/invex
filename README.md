# Invex

Missile trajectory simulation, prediction, and interception platform.

## Overview

Invex simulates realistic missile paths over procedurally generated terrain, with sensors and radars providing partial observations. The long-term goal is an adversarial reinforcement-learning framework where launcher and interceptor agents compete against each other.

## Goals

1. **Terrain & Trajectory Simulation** — Simulate missile paths over terrain with physics (gravity, drag, thrust profiles, terrain elevation).
2. **Trajectory Prediction** — From partial sensor observations, predict missile origin and destination.
3. **Interceptor Design** — Design and optimize interceptor missiles to neutralize incoming threats.
4. **Adversarial RL** — Competing launcher and interceptor learning agents.

## Project Structure

| File | Description |
|---|---|
| `terrain.py` | Procedural terrain generation (mountains, ridges, fractal noise), site placement (launch site, targets), sensor/radar placement with sweeping detection cones, line-of-sight queries, and 2D/3D visualization. |
| `trajectory.py` | Missile trajectory simulation and physics. |
| `scheme.py` | Centralized visual theme — color palette, typography, plot helpers, and glow effects for all renders. |

## Key Concepts

- **Terrain** — A 50x50 km elevation grid (configurable) with Gaussian mountains, elongated ridges, and fractal noise for roughness. Launch site is placed in the south, targets in the north.
- **Sensors** — Short-range pulsing sweep detectors with narrow beams that rotate over time. Only detect objects above terrain.
- **Radars** — Long-range, slow-sweep detectors with wider beams.
- **Detection Feed** — Aggregates all sensor/radar readings into a unified per-timestep snapshot, producing either a dict or a NumPy array of detections.
- **Coordinate System** — Right-handed ENU: x = east, y = north, z = up. All values in SI units (meters, seconds, kilograms).

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`

## Usage

Generate and visualize terrain:

```bash
python terrain.py
```

This produces a side-by-side 2D elevation map and 3D surface plot, saved to `terrain_preview.png`.

## Development

```bash
python -m pytest tests/
```

## License

Private.
