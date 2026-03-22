"""
Missile physics for Invex.
━━━━━━━━━━━━━━━━━━━━━━━━━
ENU coordinates (x=east, y=north, z=up), SI units.
"""

import numpy as np
from dataclasses import dataclass

G = 9.81  # m/s²


@dataclass
class MissileConfig:
    """Physical and operational limits for a missile."""
    max_speed: float = 600.0        # m/s  (~Mach 1.75)
    max_altitude: float = 15_000.0  # m ASL
    max_g: float = 20.0             # peak command acceleration (g)
    dt: float = 2.0                 # simulation timestep (s)
    drag_coeff: float = 5e-5        # aero drag constant (F = k·|v|·v); ≈0.5g at 300 m/s
    target_radius: float = 500.0    # arrival threshold (m)


@dataclass
class MissileState:
    """Instantaneous missile state."""
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    t: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


def step_missile(state: MissileState, accel: np.ndarray,
                 config: MissileConfig) -> MissileState:
    """Advance one timestep given a 3-D acceleration command.

    Parameters
    ----------
    state  : current missile state
    accel  : desired acceleration [ax, ay, az] in m/s²;
             magnitude is clipped to config.max_g * G
    config : missile physical limits

    Physics applied (in order):
      1. Clip command to max g-force.
      2. Add gravity.
      3. Apply linear aero drag.
      4. Euler-integrate velocity, then clamp to max_speed.
      5. Euler-integrate position.
    """
    dt = config.dt
    max_a = config.max_g * G

    a_cmd = np.asarray(accel, dtype=float)
    mag = np.linalg.norm(a_cmd)
    if mag > max_a:
        a_cmd = a_cmd * (max_a / mag)

    vel = state.vel
    drag = config.drag_coeff * np.linalg.norm(vel) * vel

    net_a = a_cmd + np.array([0.0, 0.0, -G]) - drag

    new_vel = vel + net_a * dt
    new_speed = np.linalg.norm(new_vel)
    if new_speed > config.max_speed:
        new_vel *= config.max_speed / new_speed

    new_pos = state.pos + new_vel * dt

    return MissileState(
        x=new_pos[0], y=new_pos[1], z=new_pos[2],
        vx=new_vel[0], vy=new_vel[1], vz=new_vel[2],
        t=state.t + dt,
    )
