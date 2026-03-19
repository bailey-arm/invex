"""
Terrain generation for Invex.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a 2D elevation map with mountains, valleys, target areas,
and a launch site. All coordinates in meters (SI / ENU).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Site:
    """A named location on the terrain (launch site or target)."""
    name: str
    x: float          # meters east
    y: float          # meters north
    radius: float     # area radius in meters


@dataclass
class Sensor:
    """A pulsing sweep sensor with a narrow detection cone."""
    name: str
    x: float              # position east (m)
    y: float              # position north (m)
    radius: float         # detection range (m)
    half_angle: float     # half-width of the beam (degrees)
    heading: float        # initial sweep direction (degrees, 0 = east, CCW)
    pulse_speed: float    # sweep rotation speed (degrees/s)

    def cone_at(self, t: float) -> tuple[float, float]:
        """Return (center_angle_deg, half_angle_deg) of the beam at time *t*."""
        angle = (self.heading + self.pulse_speed * t) % 360.0
        return angle, self.half_angle

    def detects(self, t: float, px: float, py: float, pz: float,
                terrain_elevation: float) -> bool:
        """True if point (px, py, pz) is inside the sensor cone at time *t*.

        Only detects objects above the terrain surface.
        """
        if pz <= terrain_elevation:
            return False

        dx = px - self.x
        dy = py - self.y
        dist = np.hypot(dx, dy)
        if dist > self.radius or dist < 1e-6:
            return False

        bearing = np.degrees(np.arctan2(dy, dx)) % 360.0
        center, half = self.cone_at(t)
        delta = (bearing - center + 180.0) % 360.0 - 180.0  # signed diff
        return abs(delta) <= half


@dataclass
class TerrainConfig:
    """Parameters for terrain generation."""
    # Grid
    size_x: float = 50_000.0          # east-west extent (m)
    size_y: float = 50_000.0          # north-south extent (m)
    resolution: float = 100.0         # grid cell size (m)

    # Elevation
    base_elevation: float = 200.0     # floor elevation (m)
    num_mountains: int = 12
    mountain_height_range: tuple = (800.0, 3500.0)
    mountain_spread_range: tuple = (1500.0, 5000.0)

    # Ridges — elongated mountain chains
    num_ridges: int = 1
    ridge_height_range: tuple = (600.0, 2000.0)
    ridge_length_range: tuple = (10_000.0, 25_000.0)
    ridge_width_range: tuple = (1500.0, 3000.0)

    # Perlin-style noise octaves for roughness
    noise_octaves: int = 5
    noise_scale: float = 150.0       # amplitude of finest detail (m)

    # Sites
    num_targets: int = 4
    target_radius: float = 500.0     # meters
    launch_radius: float = 300.0
    
    # Sensors — fast pulse, short range, narrow beam
    num_sensors: int = 3
    sensor_radius_range: tuple = (100.0, 1000.0)        # detection range (m)
    sensor_angle_range: tuple = (10.0, 45.0)            # beam half-angle (deg)
    sensor_pulse_speed: float = 5.0                     # sweep speed (deg/s)

    # Radars — slow pulse, long range, wide beam
    num_radars: int = 2
    radar_radius_range: tuple = (8000.0, 20000.0)       # detection range (m)
    radar_angle_range: tuple = (15.0, 30.0)             # beam half-angle (deg)
    radar_pulse_speed: float = 0.5                      # sweep speed (deg/s)


class Terrain:
    """Generated terrain with elevation grid, targets, and launch site."""

    def __init__(self, config: TerrainConfig | None = None, seed: int = 42):
        self.config = config or TerrainConfig()
        self.rng = np.random.default_rng(seed)

        c = self.config
        self.nx = int(c.size_x / c.resolution)
        self.ny = int(c.size_y / c.resolution)
        self.xs = np.linspace(0, c.size_x, self.nx)
        self.ys = np.linspace(0, c.size_y, self.ny)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)

        self.elevation = self._generate_elevation()
        self.launch_site = self._place_launch_site()
        self.targets = self._place_targets()
        self.sensors = self._place_sensors()
        self.radars = self._place_radars()

    # ── Elevation Generation ───────────────────────────────────────────────

    def _generate_elevation(self) -> np.ndarray:
        c = self.config
        z = np.full_like(self.grid_x, c.base_elevation)

        # Mountains — radial Gaussians
        for _ in range(c.num_mountains):
            cx = self.rng.uniform(0, c.size_x)
            cy = self.rng.uniform(0, c.size_y)
            h = self.rng.uniform(*c.mountain_height_range)
            s = self.rng.uniform(*c.mountain_spread_range)
            z += h * np.exp(-((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2) / (2 * s ** 2))

        # Ridges — elongated Gaussians rotated at random angles
        for _ in range(c.num_ridges):
            cx = self.rng.uniform(0, c.size_x)
            cy = self.rng.uniform(0, c.size_y)
            h = self.rng.uniform(*c.ridge_height_range)
            length = self.rng.uniform(*c.ridge_length_range)
            width = self.rng.uniform(*c.ridge_width_range)
            angle = self.rng.uniform(0, np.pi)

            dx = self.grid_x - cx
            dy = self.grid_y - cy
            along = dx * np.cos(angle) + dy * np.sin(angle)
            across = -dx * np.sin(angle) + dy * np.cos(angle)

            z += h * np.exp(-(along ** 2) / (2 * length ** 2)
                            - (across ** 2) / (2 * width ** 2))

        # Fractal noise for roughness
        z += self._fractal_noise()

        # Clamp floor
        z = np.maximum(z, 0.0)
        return z

    def _fractal_noise(self) -> np.ndarray:
        """Layered sinusoidal noise (cheap stand-in for Perlin)."""
        c = self.config
        noise = np.zeros_like(self.grid_x)
        amplitude = c.noise_scale
        for octave in range(c.noise_octaves):
            freq = 2 ** octave / c.size_x
            px = self.rng.uniform(0, 1000)
            py = self.rng.uniform(0, 1000)
            noise += amplitude * np.sin(2 * np.pi * freq * self.grid_x + px) \
                              * np.cos(2 * np.pi * freq * self.grid_y + py)
            amplitude *= 0.5
        return noise

    # ── Site Placement ─────────────────────────────────────────────────────

    def _place_launch_site(self) -> Site:
        """Place the launch site on the southern edge of the map, in a valley."""
        c = self.config
        margin = c.size_x * 0.15
        x = self.rng.uniform(margin, c.size_x - margin)
        y = self.rng.uniform(c.size_y * 0.05, c.size_y * 0.15)
        # Flatten terrain around launch site
        self._flatten_area(x, y, c.launch_radius * 3)
        return Site("LAUNCH", x, y, c.launch_radius)

    def _place_targets(self) -> list[Site]:
        """Place N targets in the northern half, spread apart, in clearings."""
        c = self.config
        targets = []
        min_separation = c.size_x * 0.15

        for i in range(c.num_targets):
            for _attempt in range(200):
                x = self.rng.uniform(c.size_x * 0.1, c.size_x * 0.9)
                y = self.rng.uniform(c.size_y * 0.5, c.size_y * 0.9)
                if all(np.hypot(x - t.x, y - t.y) > min_separation for t in targets):
                    break

            self._flatten_area(x, y, c.target_radius * 2)
            targets.append(Site(f"TGT-{i + 1}", x, y, c.target_radius))

        return targets

    def _place_sensors(self) -> list[Sensor]:
        """Place sensors spread across the terrain with random headings."""
        c = self.config
        sensors = []
        min_separation = c.size_x * 0.10

        for i in range(c.num_sensors):
            for _attempt in range(200):
                x = self.rng.uniform(c.size_x * 0.1, c.size_x * 0.9)
                y = self.rng.uniform(c.size_y * 0.2, c.size_y * 0.8)
                if all(np.hypot(x - s.x, y - s.y) > min_separation for s in sensors):
                    break

            radius = self.rng.uniform(*c.sensor_radius_range)
            half_angle = self.rng.uniform(*c.sensor_angle_range)
            heading = self.rng.uniform(0, 360)  # random direction, not toward launch
            sensors.append(Sensor(
                name=f"SNS-{i + 1}",
                x=x, y=y,
                radius=radius,
                half_angle=half_angle,
                heading=heading,
                pulse_speed=c.sensor_pulse_speed,
            ))

        return sensors

    def _place_radars(self) -> list[Sensor]:
        """Place radar stations — large radius, slow sweep."""
        c = self.config
        radars = []
        min_separation = c.size_x * 0.20

        for i in range(c.num_radars):
            for _attempt in range(200):
                x = self.rng.uniform(c.size_x * 0.15, c.size_x * 0.85)
                y = self.rng.uniform(c.size_y * 0.3, c.size_y * 0.7)
                if all(np.hypot(x - r.x, y - r.y) > min_separation for r in radars):
                    break

            radius = self.rng.uniform(*c.radar_radius_range)
            half_angle = self.rng.uniform(*c.radar_angle_range)
            heading = self.rng.uniform(0, 360)
            radars.append(Sensor(
                name=f"RDR-{i + 1}",
                x=x, y=y,
                radius=radius,
                half_angle=half_angle,
                heading=heading,
                pulse_speed=c.radar_pulse_speed,
            ))

        return radars

    def _flatten_area(self, cx: float, cy: float, radius: float):
        """Smooth the elevation to a flat pad around a point."""
        dist = np.hypot(self.grid_x - cx, self.grid_y - cy)
        mask = dist < radius
        if not mask.any():
            return
        center_elev = self.elevation[mask].min()
        blend = np.clip(1.0 - dist / radius, 0, 1) ** 2
        self.elevation = self.elevation * (1 - blend) + center_elev * blend

    # ── Queries ────────────────────────────────────────────────────────────

    def elevation_at(self, x: float, y: float) -> float:
        """Interpolated elevation at an arbitrary (x, y) point."""
        c = self.config
        ix = np.clip(x / c.resolution, 0, self.nx - 1)
        iy = np.clip(y / c.resolution, 0, self.ny - 1)

        ix0, iy0 = int(ix), int(iy)
        ix1 = min(ix0 + 1, self.nx - 1)
        iy1 = min(iy0 + 1, self.ny - 1)
        fx, fy = ix - ix0, iy - iy0

        return float(
            self.elevation[iy0, ix0] * (1 - fx) * (1 - fy)
            + self.elevation[iy0, ix1] * fx * (1 - fy)
            + self.elevation[iy1, ix0] * (1 - fx) * fy
            + self.elevation[iy1, ix1] * fx * fy
        )

    def line_of_sight(self, x0, y0, z0, x1, y1, z1, steps=200) -> bool:
        """True if a straight line between two 3D points clears the terrain."""
        ts = np.linspace(0, 1, steps)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        zs = z0 + (z1 - z0) * ts
        for x, y, z in zip(xs, ys, zs):
            if z <= self.elevation_at(x, y):
                return False
        return True

    # ── Visualization ──────────────────────────────────────────────────────

    def plot(self, ax=None, show_sites=True):
        """Render the terrain as a top-down elevation map."""
        import matplotlib.pyplot as plt
        import scheme

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.get_figure()

        scheme.apply_theme(ax, title="TERRAIN MAP")

        cmap = scheme.terrain_cmap()
        im = ax.pcolormesh(
            self.grid_x / 1000, self.grid_y / 1000, self.elevation,
            cmap=cmap, shading="gouraud",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Elevation (m)", color=scheme.TEXT_SECONDARY, fontsize=scheme.FONT_SIZE_LABEL)
        cbar.ax.tick_params(colors=scheme.TEXT_SECONDARY, labelsize=scheme.FONT_SIZE_TICK)
        cbar.outline.set_edgecolor(scheme.GRID)

        if show_sites:
            ls = self.launch_site
            scheme.point(ax, ls.x / 1000, ls.y / 1000, color=scheme.AMBER, label=ls.name)

            for tgt in self.targets:
                scheme.point(ax, tgt.x / 1000, tgt.y / 1000, color=scheme.MAGENTA, label=tgt.name)
                circle = plt.Circle(
                    (tgt.x / 1000, tgt.y / 1000), tgt.radius / 1000,
                    fill=False, edgecolor=scheme.MAGENTA, linewidth=0.8, alpha=0.6,
                )
                ax.add_patch(circle)

            # Radars — show position and initial sweep cone
            from matplotlib.patches import Wedge
            for rdr in self.radars:
                scheme.point(ax, rdr.x / 1000, rdr.y / 1000,
                             color=scheme.CYAN, label=rdr.name)
                theta1 = rdr.heading - rdr.half_angle
                theta2 = rdr.heading + rdr.half_angle
                wedge = Wedge(
                    (rdr.x / 1000, rdr.y / 1000),
                    rdr.radius / 1000,
                    theta1, theta2,
                    facecolor=scheme.CYAN_GLOW,
                    edgecolor=scheme.CYAN,
                    linewidth=0.8, alpha=0.3,
                )
                ax.add_patch(wedge)

            # Sensors — show position and initial sweep cone
            for sns in self.sensors:
                scheme.point(ax, sns.x / 1000, sns.y / 1000,
                             color=scheme.ELECTRIC_BLUE, label=sns.name)
                theta1 = sns.heading - sns.half_angle
                theta2 = sns.heading + sns.half_angle
                wedge = Wedge(
                    (sns.x / 1000, sns.y / 1000),
                    sns.radius / 1000,
                    theta1, theta2,
                    facecolor=scheme.ELECTRIC_BLUE_GLOW,
                    edgecolor=scheme.ELECTRIC_BLUE,
                    linewidth=0.8, alpha=0.5,
                )
                ax.add_patch(wedge)

            ax.legend(
                loc="upper right", fontsize=scheme.FONT_SIZE_LABEL,
                facecolor=scheme.SURFACE, edgecolor=scheme.GRID,
                labelcolor=scheme.TEXT_PRIMARY,
            )

        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_aspect("equal")
        return ax

    def plot_3d(self, ax=None):
        """Render the terrain as a 3D surface."""
        import matplotlib.pyplot as plt
        import scheme

        if ax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")

        scheme.apply_theme_3d(ax, title="TERRAIN — 3D")

        cmap = scheme.terrain_cmap()
        ax.plot_surface(
            self.grid_x / 1000, self.grid_y / 1000, self.elevation,
            cmap=cmap, edgecolor=scheme.TERRAIN_EDGE, linewidth=0.1, alpha=0.9,
        )

        ls = self.launch_site
        lz = self.elevation_at(ls.x, ls.y)
        ax.scatter(ls.x / 1000, ls.y / 1000, lz + 50, color=scheme.AMBER, s=60, zorder=5)

        for tgt in self.targets:
            tz = self.elevation_at(tgt.x, tgt.y)
            ax.scatter(tgt.x / 1000, tgt.y / 1000, tz + 50, color=scheme.MAGENTA, s=60, zorder=5)

        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_zlabel("Elevation (m)")
        return ax


class DetectionFeed:
    """Integrated detection feed across all sensors and radars.

    At each time step, queries every sensor/radar and produces a row with
    one column per detector.  Each column holds either the detected (x, y, z)
    position or NaN if the detector saw nothing at that instant.
    """

    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.detectors: list[Sensor] = terrain.sensors + terrain.radars
        self.columns: list[str] = [d.name for d in self.detectors]

    def query(self, t: float, missiles: list[tuple[float, float, float]]
              ) -> dict[str, tuple[float, float, float] | None]:
        """Return one snapshot of the feed at time *t*.

        Parameters
        ----------
        t : float
            Simulation time in seconds.
        missiles : list of (x, y, z)
            Current positions of all airborne missiles.

        Returns
        -------
        dict mapping detector name → (x, y, z) of the first missile it
        detects, or None if it sees nothing.
        """
        row: dict[str, tuple[float, float, float] | None] = {
            name: None for name in self.columns
        }

        for det in self.detectors:
            for mx, my, mz in missiles:
                elev = self.terrain.elevation_at(mx, my)
                if det.detects(t, mx, my, mz, elev):
                    row[det.name] = (mx, my, mz)
                    break  # one detection per detector per tick

        return row

    def query_array(self, t: float, missiles: list[tuple[float, float, float]]
                    ) -> np.ndarray:
        """Same as *query* but returns a (N_detectors, 3) float array.

        Undetected entries are filled with NaN.
        """
        row = self.query(t, missiles)
        out = np.full((len(self.detectors), 3), np.nan)
        for i, name in enumerate(self.columns):
            if row[name] is not None:
                out[i] = row[name]
        return out


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    terrain = Terrain()

    print(f"Grid: {terrain.nx} x {terrain.ny}")
    print(f"Elevation range: {terrain.elevation.min():.0f} – {terrain.elevation.max():.0f} m")
    print(f"Launch: {terrain.launch_site}")
    for t in terrain.targets:
        print(f"Target: {t}")
    for s in terrain.sensors:
        print(f"Sensor: {s}")
    for r in terrain.radars:
        print(f"Radar:  {r}")

    # Quick detection feed demo — missile at map center, 1 km altitude
    feed = DetectionFeed(terrain)
    missile_pos = [(25000.0, 25000.0, 1000.0)]
    for t in range(0, 30, 5):
        row = feed.query(float(t), missile_pos)
        hits = {k: v for k, v in row.items() if v is not None}
        print(f"  t={t:3d}s  detections: {hits if hits else '—'}")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    terrain.plot(ax=axes[0])

    axes[1].remove()
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    terrain.plot_3d(ax=ax3d)

    plt.tight_layout()
    plt.savefig("terrain_preview.png", dpi=150, facecolor=fig.get_facecolor())
    plt.show()
