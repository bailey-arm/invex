"""
NavigationEnv — RL environment for missile terrain-avoidance navigation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gym-compatible interface (reset / step / render), no gym dependency.

Observation (24-D, all values normalised to ≈ [-1, 1]):
  [0:3]   position / [size_x, size_y, max_alt]
  [3:6]   velocity / max_speed
  [6:9]   vector to target / initial_dist
  [9]     terrain clearance at current position / max_alt
  [10:20] terrain clearance at 10 fan probes / max_alt
            Fan at look-ahead time T = 10 s:
            angles [-90°, -60°, -30°, 0°, 30°, 60°, 90°] (7 probes)
            Plus straight-ahead at T = [3, 8, 20] s  (3 probes)
  [20]    current speed / max_speed
  [21:24] unit vector toward target

Action (3-D, each in [-1, 1]):
  Desired acceleration [ax, ay, az]; scaled by max_g * G inside step().
"""

import numpy as np
from trajectory import MissileConfig, MissileState, step_missile, G

# Fan probe angles (degrees from velocity direction, in horizontal plane)
_FAN_ANGLES_DEG = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
_FAN_LOOK_T     = 10.0   # seconds ahead for the fan

# Additional straight-ahead probes at different distances
_STRAIGHT_LOOK_T = [3.0, 8.0, 20.0]

# Terrain clearance below which a continuous penalty kicks in (m)
_SAFE_CLEARANCE = 250.0


class NavigationEnv:
    """Navigate a missile from start to target while staying above terrain."""

    OBS_DIM = 24   # 3+3+3+1+10+1+3
    ACT_DIM = 3

    def __init__(self, terrain, config: MissileConfig = None,
                 max_steps: int = 600):
        self.terrain = terrain
        self.config = config or MissileConfig()
        self.max_steps = max_steps

        self._state: MissileState = None
        self._target: np.ndarray = None
        self._initial_dist: float = None
        self._step_count: int = 0

        c = terrain.config
        self._map_size = np.array([c.size_x, c.size_y, self.config.max_altitude])

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, start: np.ndarray, target: np.ndarray,
              initial_speed: float = 150.0) -> np.ndarray:
        """Reset episode and return the initial observation.

        Parameters
        ----------
        start, target : (x, y, z) positions in metres (ENU).
        initial_speed : launch speed directed toward the target (m/s).
        """
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)

        # Direct initial velocity toward target in XY (horizontal guidance).
        # The agent controls altitude; we start purely horizontal.
        horiz = target[:2] - start[:2]
        horiz_dist = np.linalg.norm(horiz)
        horiz_unit = horiz / max(horiz_dist, 1.0)
        v0 = np.array([horiz_unit[0] * initial_speed,
                        horiz_unit[1] * initial_speed,
                        0.0])

        self._state = MissileState(
            x=start[0], y=start[1], z=start[2],
            vx=v0[0], vy=v0[1], vz=v0[2],
        )
        self._target      = target
        self._cruise_z    = float(start[2])   # maintain this altitude throughout
        dist_2d = float(np.linalg.norm((target - start)[:2]))
        self._initial_dist = max(dist_2d, 1.0)
        self._step_count = 0

        return self._obs()

    def step(self, action: np.ndarray):
        """Advance one timestep.

        Parameters
        ----------
        action : np.ndarray (3,) in [-1, 1], scaled internally to m/s².

        Returns
        -------
        obs, reward, done, info
        """
        action = np.clip(action, -1.0, 1.0)
        accel = action * self.config.max_g * G

        prev_state = self._state
        self._state = step_missile(self._state, accel, self.config)
        self._step_count += 1

        pos = self._state.pos
        terrain_z = self.terrain.elevation_at(pos[0], pos[1])
        clearance = pos[2] - terrain_z

        hit_terrain = clearance <= 0.0
        out_of_bounds = (
            pos[0] < 0 or pos[0] > self.terrain.config.size_x or
            pos[1] < 0 or pos[1] > self.terrain.config.size_y
        )
        above_ceiling = pos[2] > self.config.max_altitude
        # Arrival is checked in 2-D (XY only) — altitude is handled by terrain
        # avoidance; terminal descent is a separate phase.
        dist_to_target = float(np.linalg.norm(pos - self._target))
        dist_2d        = float(np.linalg.norm(pos[:2] - self._target[:2]))
        arrived = dist_2d <= self.config.target_radius
        timeout = self._step_count >= self.max_steps

        done = hit_terrain or out_of_bounds or above_ceiling or arrived or timeout
        reward = self._reward(prev_state, clearance, hit_terrain,
                              out_of_bounds, above_ceiling, arrived,
                              dist_2d)

        info = {
            "arrived":        arrived,
            "hit_terrain":    hit_terrain,
            "out_of_bounds":  out_of_bounds,
            "timeout":        timeout,
            "dist_to_target": dist_2d,
            "clearance":      clearance,
            "t":              self._state.t,
        }
        return self._obs(), reward, done, info

    # ── Internals ────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        s = self._state
        pos = s.pos
        vel = s.vel
        speed = max(s.speed, 1.0)

        terrain_z = self.terrain.elevation_at(pos[0], pos[1])
        clearance = (pos[2] - terrain_z) / self.config.max_altitude

        to_target = self._target - pos
        dist = np.linalg.norm(to_target)
        target_dir = to_target / max(dist, 1.0)

        # ── Terrain probes ───────────────────────────────────────────────────
        # Horizontal velocity direction (used to orient the fan)
        vel_horiz = np.array([vel[0], vel[1], 0.0])
        h_mag = np.linalg.norm(vel_horiz)
        if h_mag > 0.1:
            h_dir = vel_horiz / h_mag
        else:
            # Fall back to target direction when nearly stationary
            td = np.array([target_dir[0], target_dir[1], 0.0])
            td_mag = np.linalg.norm(td)
            h_dir = td / max(td_mag, 1e-6)

        def _probe_clearance(angle_deg: float, look_t: float) -> float:
            """Clearance at a probe point (angle_deg off h_dir, look_t seconds away)."""
            a = np.radians(angle_deg)
            ca, sa = np.cos(a), np.sin(a)
            fan_dir = np.array([h_dir[0] * ca - h_dir[1] * sa,
                                h_dir[0] * sa + h_dir[1] * ca,
                                0.0])
            probe = pos + fan_dir * speed * look_t
            px = float(np.clip(probe[0], 0, self.terrain.config.size_x))
            py = float(np.clip(probe[1], 0, self.terrain.config.size_y))
            t_z = self.terrain.elevation_at(px, py)
            lc = (pos[2] - t_z) / self.config.max_altitude   # clearance at cruise alt
            return float(np.clip(lc, -1.0, 1.0))

        # Fan: 7 directions at _FAN_LOOK_T
        fan_clearances = [_probe_clearance(a, _FAN_LOOK_T)
                          for a in _FAN_ANGLES_DEG]
        # Straight: 3 distances
        straight_clearances = [_probe_clearance(0.0, lt)
                                for lt in _STRAIGHT_LOOK_T]

        obs = np.concatenate([
            pos / self._map_size,                              # 3
            vel / self.config.max_speed,                       # 3
            to_target / self._initial_dist,                    # 3
            [float(np.clip(clearance, -1.0, 1.0))],            # 1
            fan_clearances,                                    # 7
            straight_clearances,                               # 3
            [speed / self.config.max_speed],                   # 1
            target_dir,                                        # 3
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float64)

    def _reward(self, prev_state: MissileState, clearance: float,
                hit_terrain: bool, out_of_bounds: bool,
                above_ceiling: bool, arrived: bool,
                dist_to_target: float) -> float:
        if arrived:
            return 100.0
        if hit_terrain or out_of_bounds or above_ceiling:
            # All hard terminations are equally bad — don't reward climbing to ceiling
            return -100.0

        pos = self._state.pos

        # Progress toward target in 2-D (lateral guidance only)
        prev_dist_2d = float(np.linalg.norm(prev_state.pos[:2] - self._target[:2]))
        progress = (prev_dist_2d - dist_to_target) / self._initial_dist

        # Terrain clearance penalty (quadratic from 0 at _SAFE_CLEARANCE to -3 at 0m)
        if clearance < _SAFE_CLEARANCE:
            terrain_penalty = -3.0 * (1.0 - clearance / _SAFE_CLEARANCE) ** 2
        else:
            terrain_penalty = 0.0

        return progress * 10.0 + terrain_penalty - 0.02

    # ── Visualisation ────────────────────────────────────────────────────────

    def render_trajectory(self, positions: list, target: np.ndarray,
                          ax=None, label: str = ""):
        """Overlay a trajectory (list of (x,y,z)) on a 2-D terrain map."""
        import matplotlib.pyplot as plt
        import scheme

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
            self.terrain.plot(ax=ax)

        xs = [p[0] / 1000 for p in positions]
        ys = [p[1] / 1000 for p in positions]
        ax.plot(xs, ys, color=scheme.AMBER, linewidth=1.2,
                alpha=0.85, label=label or "trajectory")
        ax.scatter(xs[0], ys[0], color=scheme.ELECTRIC_BLUE, s=40, zorder=5)
        ax.scatter(xs[-1], ys[-1], color=scheme.MAGENTA, s=40, zorder=5)
        return ax
