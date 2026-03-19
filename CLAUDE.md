# Invex

Missile trajectory simulation, prediction, and interception platform.

## Project Goals

1. **Terrain & Trajectory Simulation** — Simulate realistic missile paths over terrain, accounting for physics (gravity, drag, thrust profiles, terrain elevation).
2. **Trajectory Prediction** — Given partial observations of a missile in flight, predict:
   - **Origin** — where it was launched from.
   - **Destination** — where it is heading.
3. **Interceptor Design** — Design and optimize interceptor missiles that neutralize incoming threats.
4. **Adversarial RL** — End-state architecture is a reinforcement-learning framework where both the launcher and interceptor are learning agents competing against each other.

## Architecture Overview

- `terrain.py` — Terrain generation and representation.
- `trajectory.py` — Missile trajectory simulation and physics.

## Tech Stack

- Python

## Conventions

- Keep physics and RL logic in separate modules; simulation should be usable without the RL layer.
- Use SI units (meters, seconds, kilograms) throughout.
- Coordinate system: x = east, y = north, z = up (right-handed ENU).

## Development

```bash
# Run from project root
python -m pytest tests/
```
