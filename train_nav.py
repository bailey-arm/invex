"""
Train and evaluate the missile navigation agent.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
    python train_nav.py                         # 1 000 episodes, all targets
    python train_nav.py --episodes 3000 --target 2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

from terrain import Terrain
from trajectory import MissileConfig
from nav_env import NavigationEnv
from nav_agent import REINFORCEAgent
import scheme


# ── Helpers ───────────────────────────────────────────────────────────────────

def _launch_z(terrain: Terrain) -> float:
    """Cruise altitude: 60th-percentile terrain elevation + 150 m.

    This puts the cruise altitude below the tallest peaks so high terrain
    genuinely blocks the straight-line PN path and forces the RL agent to
    learn lateral deviations around obstacles.
    """
    return float(np.percentile(terrain.elevation, 85)) + 150.0


def collect_episode(env: NavigationEnv, agent: REINFORCEAgent,
                    start: np.ndarray, target: np.ndarray,
                    greedy: bool = False,
                    initial_speed: float = 300.0) -> dict:
    """Run one episode and return trajectory data."""
    obs = env.reset(start, target, initial_speed=initial_speed)
    obs_list, act_list, rew_list, pos_list = [], [], [], []
    done = False

    while not done:
        pos_list.append(env._state.pos.copy())
        obs_list.append(obs)          # store obs BEFORE the action
        act = agent.act(obs, deterministic=greedy)
        obs, rew, done, info = env.step(act)
        act_list.append(act)
        rew_list.append(rew)

    pos_list.append(env._state.pos.copy())

    return {
        "obs":          np.array(obs_list),   # (T, obs_dim) — aligned with actions
        "actions":      np.array(act_list),
        "rewards":      rew_list,
        "positions":    pos_list,
        "info":         info,
        "total_reward": float(np.sum(rew_list)),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    n_episodes: int = 1000,
    batch_size: int = 16,
    seed: int = 42,
    snapshot_every: int = 50,   # greedy rollout every N *training episodes*
    target_idx: int = None,     # None = random per episode; int = fixed target
    verbose: bool = True,
) -> dict:
    """Train the navigation agent.

    Returns
    -------
    dict with keys:
        agent, env, terrain, history, snapshots
    """
    rng = np.random.default_rng(seed)
    terrain = Terrain(seed=seed)

    missile_cfg = MissileConfig(
        max_speed=500.0,
        max_altitude=12_000.0,
        max_g=15.0,
        dt=3.0,
        target_radius=4000.0,   # 2-D arrival radius (m); wide enough for early learning
    )
    env   = NavigationEnv(terrain, missile_cfg, max_steps=400)
    agent = REINFORCEAgent(
        obs_dim=NavigationEnv.OBS_DIM,
        act_dim=NavigationEnv.ACT_DIM,
        hidden=64,
        lr_policy=1e-4,
        lr_value=5e-4,
        gamma=0.99,
        init_log_std=-0.5,    # std≈0.61: wide initial exploration, visibly diverse paths
        entropy_coeff=0.001,  # low: lets std decay as agent finds good routes
        seed=seed,
    )

    ls      = terrain.launch_site
    sz      = _launch_z(terrain)
    targets = terrain.targets

    # Choose a fixed snapshot target (default: target 0)
    snap_tgt_idx = 0 if target_idx is None else target_idx
    snap_tgt     = targets[snap_tgt_idx]
    snap_start   = np.array([ls.x, ls.y, sz])
    tz           = sz
    snap_target  = np.array([snap_tgt.x, snap_tgt.y, tz])

    history = {
        "episode":      [],
        "mean_reward":  [],
        "arrival_rate": [],
    }

    # snapshots[i] = {"episode": int, "positions": list, "arrived": bool}
    snapshots: list = []

    ep           = 0
    update_count = 0
    last_snap_ep = -snapshot_every  # trigger first snapshot before episode 0

    while ep < n_episodes:

        # ── Snapshot before this batch ──────────────────────────────────────
        if ep - last_snap_ep >= snapshot_every:
            snap = collect_episode(env, agent, snap_start, snap_target,
                                   greedy=True, initial_speed=300.0)
            # Collect stochastic samples to show exploration spread in GIF
            samples = [collect_episode(env, agent, snap_start, snap_target,
                                       greedy=False, initial_speed=300.0)
                       for _ in range(16)]
            snapshots.append({
                "episode":   ep,
                "positions": snap["positions"],
                "arrived":   snap["info"]["arrived"],
                "reward":    snap["total_reward"],
                "samples":   samples,
            })
            last_snap_ep = ep

        # ── Collect a training batch ─────────────────────────────────────
        batch    = []
        arrivals = 0

        for _ in range(batch_size):
            if target_idx is None:
                tgt = targets[int(rng.integers(len(targets)))]
            else:
                tgt = targets[target_idx]

            start  = np.array([ls.x, ls.y, sz])
            target = np.array([tgt.x, tgt.y, sz])

            traj = collect_episode(env, agent, start, target,
                                   initial_speed=300.0)
            batch.append(traj)
            if traj["info"]["arrived"]:
                arrivals += 1
            ep += 1

        agent.update(batch)
        update_count += 1

        mean_rew    = float(np.mean([t["total_reward"] for t in batch]))
        arrival_pct = arrivals / batch_size * 100.0

        history["episode"].append(ep)
        history["mean_reward"].append(mean_rew)
        history["arrival_rate"].append(arrival_pct)

        if verbose and update_count % 10 == 0:
            print(
                f"ep {ep:5d}  reward {mean_rew:+8.2f}  "
                f"arrived {arrival_pct:5.1f}%"
            )

    # Final snapshot
    snap = collect_episode(env, agent, snap_start, snap_target,
                           greedy=True, initial_speed=300.0)
    samples = [collect_episode(env, agent, snap_start, snap_target,
                               greedy=False, initial_speed=300.0)
               for _ in range(16)]
    snapshots.append({
        "episode":   ep,
        "positions": snap["positions"],
        "arrived":   snap["info"]["arrived"],
        "reward":    snap["total_reward"],
        "samples":   samples,
    })

    return {
        "agent":       agent,
        "env":         env,
        "terrain":     terrain,
        "history":     history,
        "snapshots":   snapshots,
        "snap_target": snap_tgt,
        "snap_start":  snap_start,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_generations(result: dict, ax=None) -> plt.Axes:
    """Overlay all generation snapshots on the terrain map.

    Lines fade from dim-blue (early) to bright-amber (late).
    A solid magenta line marks the final best run.
    """
    terrain   = result["terrain"]
    snapshots = result["snapshots"]
    snap_tgt  = result["snap_target"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 11))
        terrain.plot(ax=ax, show_sites=True)

    n = len(snapshots)
    cmap = plt.colormaps["plasma"]

    for i, snap in enumerate(snapshots):
        t_norm = i / max(n - 1, 1)          # 0 (first) → 1 (last)
        rgba   = cmap(t_norm)
        alpha  = 0.25 + 0.65 * t_norm       # fade in as training progresses
        lw     = 0.6 + 1.2 * t_norm

        positions = snap["positions"]
        xs = [p[0] / 1000 for p in positions]
        ys = [p[1] / 1000 for p in positions]

        is_last = (i == n - 1)
        ax.plot(xs, ys, color=rgba, linewidth=lw, alpha=alpha,
                zorder=3 + i,
                label=f"gen {snap['episode']}" if is_last else None)

    # Annotate generation numbers at sparse intervals
    step = max(1, n // 6)
    for i in range(0, n, step):
        snap = snapshots[i]
        pos  = snap["positions"]
        mid  = pos[len(pos) // 2]
        t_norm = i / max(n - 1, 1)
        rgba   = cmap(t_norm)
        ax.annotate(
            str(snap["episode"]),
            (mid[0] / 1000, mid[1] / 1000),
            fontsize=7, color=rgba, alpha=0.8,
            xytext=(4, 2), textcoords="offset points",
        )

    # Colorbar-style legend: show episode scale
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(
                                    snapshots[0]["episode"],
                                    snapshots[-1]["episode"]))
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Training episode", color=scheme.TEXT_SECONDARY,
                   fontsize=scheme.FONT_SIZE_LABEL)
    cbar.ax.tick_params(colors=scheme.TEXT_SECONDARY,
                        labelsize=scheme.FONT_SIZE_TICK)
    cbar.outline.set_edgecolor(scheme.GRID)

    scheme.apply_theme(ax, title=f"TRAJECTORY EVOLUTION → {snap_tgt.name}")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_aspect("equal")
    return ax


def plot_training(history: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(scheme.BACKGROUND)

    eps = history["episode"]

    for ax_, key, color, title in [
        (axes[0], "mean_reward",  scheme.AMBER,   "Mean Episode Reward"),
        (axes[1], "arrival_rate", scheme.MAGENTA,  "Arrival Rate (%)"),
    ]:
        scheme.apply_theme(ax_, title=title)
        ax_.plot(eps, history[key], color=color, linewidth=1.2)
        ax_.set_xlabel("Episode")

    plt.tight_layout()
    return fig


def plot_missions(result: dict, n_missions: int = 4) -> plt.Figure:
    """Run the trained agent to all targets and overlay on terrain."""
    agent   = result["agent"]
    env     = result["env"]
    terrain = result["terrain"]

    ls  = terrain.launch_site
    sz  = _launch_z(terrain)

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(scheme.BACKGROUND)
    terrain.plot(ax=ax, show_sites=True)
    scheme.apply_theme(ax, title="TRAINED TRAJECTORIES — ALL TARGETS")

    palette = [scheme.AMBER, scheme.CYAN, scheme.ELECTRIC_BLUE, scheme.MAGENTA]

    for i, tgt in enumerate(terrain.targets[:n_missions]):
        start  = np.array([ls.x, ls.y, sz])
        target = np.array([tgt.x, tgt.y, sz])
        traj   = collect_episode(env, agent, start, target, greedy=True)

        positions = traj["positions"]
        xs = [p[0] / 1000 for p in positions]
        ys = [p[1] / 1000 for p in positions]

        status = ("ARRIVED"     if traj["info"]["arrived"]    else
                  "HIT TERRAIN" if traj["info"]["hit_terrain"] else
                  "TIMEOUT")
        color = palette[i % len(palette)]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9, zorder=5,
                label=f"{tgt.name} → {status}")
        ax.scatter(xs[-1], ys[-1], color=color, s=50, zorder=6)

    ax.legend(
        loc="upper right", fontsize=scheme.FONT_SIZE_LABEL,
        facecolor=scheme.SURFACE, edgecolor=scheme.GRID,
        labelcolor=scheme.TEXT_PRIMARY,
    )
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_altitude_profile(result: dict, target_idx: int = 0) -> plt.Figure:
    """Altitude vs. distance for one mission, with terrain cross-section."""
    agent   = result["agent"]
    env     = result["env"]
    terrain = result["terrain"]

    ls  = terrain.launch_site
    sz  = _launch_z(terrain)
    tgt    = terrain.targets[target_idx]
    start  = np.array([ls.x, ls.y, sz])
    target = np.array([tgt.x, tgt.y, sz])
    traj   = collect_episode(env, agent, start, target, greedy=True)

    positions = traj["positions"]
    dists = [np.linalg.norm(np.array(p[:2]) - start[:2]) / 1000
             for p in positions]
    alts  = [p[2] for p in positions]

    n_pts = 300
    ts    = np.linspace(0, 1, n_pts)
    tx    = ls.x + (tgt.x - ls.x) * ts
    ty    = ls.y + (tgt.y - ls.y) * ts
    straight_km   = np.linalg.norm([tgt.x - ls.x, tgt.y - ls.y]) / 1000
    terrain_alts  = [terrain.elevation_at(float(x), float(y))
                     for x, y in zip(tx, ty)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(scheme.BACKGROUND)
    scheme.apply_theme(ax, title=f"ALTITUDE PROFILE → {tgt.name}")

    ax.fill_between(ts * straight_km, 0, terrain_alts,
                    color=scheme.TERRAIN_EDGE, alpha=0.5, label="Terrain")
    ax.plot(dists, alts, color=scheme.AMBER, linewidth=1.5,
            label="Missile altitude")
    ax.axhline(terrain_alts[0], color=scheme.GRID, linewidth=0.6,
               linestyle="--")

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.legend(
        facecolor=scheme.SURFACE, edgecolor=scheme.GRID,
        labelcolor=scheme.TEXT_PRIMARY,
    )
    plt.tight_layout()
    return fig


def make_gif(result: dict, path: str = "nav_generations.gif", fps: int = 60):
    """Two-panel animated GIF: top-down map (all paths) + altitude profiles.

    Each frame = one generation snapshot.
    Top panel  : terrain map with all 16 stochastic paths (colour = outcome)
                 plus the greedy path as a bright white line.
    Bottom panel: altitude vs distance for every path in that frame;
                 terrain elevation below the missile is filled in.
    """
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec

    terrain   = result["terrain"]
    snapshots = result["snapshots"]
    snap_tgt  = result["snap_target"]
    snap_start = result.get("snap_start")
    sz         = _launch_z(terrain)
    n          = len(snapshots)

    # Outcome colours
    COL_ARRIVED = scheme.CYAN
    COL_TERRAIN = scheme.MAGENTA
    COL_OTHER   = scheme.AMBER

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor(scheme.BACKGROUND)

    def _traj_color(info):
        if info["arrived"]:    return COL_ARRIVED
        if info["hit_terrain"]: return COL_TERRAIN
        return COL_OTHER

    def _draw_frame(i):
        fig.clf()
        fig.patch.set_facecolor(scheme.BACKGROUND)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.35)
        ax_map = fig.add_subplot(gs[0])
        ax_alt = fig.add_subplot(gs[1])

        snap    = snapshots[i]
        samples = snap.get("samples", [])

        # ── Top panel: terrain + trajectories ───────────────────────────────
        terrain.plot(ax=ax_map, show_sites=True)
        scheme.apply_theme(ax_map, title=f"TRAJECTORY EVOLUTION → {snap_tgt.name}")
        ax_map.set_xlabel("East (km)")
        ax_map.set_ylabel("North (km)")
        ax_map.set_aspect("equal")

        # Stochastic samples (thin, coloured by outcome)
        for traj in samples:
            color = _traj_color(traj["info"])
            pos = traj["positions"]
            ax_map.plot([p[0]/1000 for p in pos], [p[1]/1000 for p in pos],
                        color=color, linewidth=0.7, alpha=0.45, zorder=4)

        # Greedy path (bright white, thick)
        gpos = snap["positions"]
        ax_map.plot([p[0]/1000 for p in gpos], [p[1]/1000 for p in gpos],
                    color="white", linewidth=2.0, alpha=0.92, zorder=6)

        arrived_n = sum(1 for t in samples if t["info"]["arrived"])
        status    = "ARRIVED" if snap["arrived"] else "HIT TERRAIN" if not snap["arrived"] else "OTHER"
        ax_map.text(0.02, 0.97,
                    f"Episode {snap['episode']}  •  greedy: {status}"
                    f"  •  stochastic: {arrived_n}/{len(samples)} arrived",
                    transform=ax_map.transAxes,
                    fontsize=8, color=scheme.TEXT_PRIMARY, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=scheme.SURFACE,
                              alpha=0.75, edgecolor=scheme.GRID))

        # Legend patches
        from matplotlib.lines import Line2D
        ax_map.legend(handles=[
            Line2D([0],[0], color="white",       lw=2,   label="Greedy"),
            Line2D([0],[0], color=COL_ARRIVED,   lw=1.5, label="Arrived"),
            Line2D([0],[0], color=COL_TERRAIN,   lw=1.5, label="Hit terrain"),
            Line2D([0],[0], color=COL_OTHER,     lw=1.5, label="Timeout/OOB"),
        ], loc="lower right", fontsize=7,
           facecolor=scheme.SURFACE, edgecolor=scheme.GRID,
           labelcolor=scheme.TEXT_PRIMARY)

        # ── Bottom panel: altitude profiles ─────────────────────────────────
        scheme.apply_theme(ax_alt, title="ALTITUDE PROFILES")
        ax_alt.set_xlabel("Distance from launch (km)")
        ax_alt.set_ylabel("Altitude (m)")
        ax_alt.axhline(sz, color=scheme.GRID, linewidth=0.8,
                       linestyle="--", alpha=0.5, label=f"Cruise {sz:.0f} m")

        ls_pos = np.array([terrain.launch_site.x, terrain.launch_site.y])

        for traj in samples:
            color = _traj_color(traj["info"])
            pos   = traj["positions"]
            dists = [np.linalg.norm(np.array(p[:2]) - ls_pos) / 1000 for p in pos]
            alts  = [p[2] for p in pos]
            terr  = [terrain.elevation_at(float(p[0]), float(p[1])) for p in pos]
            ax_alt.plot(dists, alts,  color=color, linewidth=0.7, alpha=0.45)
            ax_alt.fill_between(dists, 0, terr, color=scheme.TERRAIN_EDGE,
                                alpha=0.06)

        # Greedy path altitude (white, thick)
        dists_g = [np.linalg.norm(np.array(p[:2]) - ls_pos) / 1000 for p in gpos]
        alts_g  = [p[2] for p in gpos]
        terr_g  = [terrain.elevation_at(float(p[0]), float(p[1])) for p in gpos]
        ax_alt.plot(dists_g, alts_g, color="white", linewidth=1.8, alpha=0.9)
        ax_alt.fill_between(dists_g, 0, terr_g, color=scheme.TERRAIN_EDGE,
                            alpha=0.35, label="Terrain below path")
        ax_alt.legend(loc="upper right", fontsize=7,
                      facecolor=scheme.SURFACE, edgecolor=scheme.GRID,
                      labelcolor=scheme.TEXT_PRIMARY)

    ani = animation.FuncAnimation(fig, _draw_frame, frames=n, repeat=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int, default=10000)
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--seed",           type=int, default=2171)
    parser.add_argument("--snapshot-every", type=int, default=100,
                        help="Capture a greedy rollout every N training episodes")
    parser.add_argument("--target",         type=int, default=None,
                        help="Fixed target index (default: random each episode)")
    args = parser.parse_args()

    print(f"Training {args.episodes} episodes "
          f"(batch={args.batch_size}, seed={args.seed}, "
          f"snapshot every {args.snapshot_every} eps) …\n")

    result = train(
        n_episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        target_idx=args.target,
    )

    print(f"\n{len(result['snapshots'])} generation snapshots captured.")

    # ── Generation evolution plot ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(13, 11))
    fig1.patch.set_facecolor(scheme.BACKGROUND)
    result["terrain"].plot(ax=ax1, show_sites=True)
    plot_generations(result, ax=ax1)
    plt.tight_layout()
    fig1.savefig("nav_generations.png", dpi=150, facecolor=fig1.get_facecolor())
    print("Saved nav_generations.png")

    # ── Training curves ─────────────────────────────────────────────────────
    fig2 = plot_training(result["history"])
    fig2.savefig("nav_training.png", dpi=150, facecolor=fig2.get_facecolor())
    print("Saved nav_training.png")

    # ── Final missions to all targets ───────────────────────────────────────
    fig3 = plot_missions(result)
    fig3.savefig("nav_missions.png", dpi=150, facecolor=fig3.get_facecolor())
    print("Saved nav_missions.png")

    # ── Altitude profile ────────────────────────────────────────────────────
    snap_idx = result["snap_target"].name.split("-")[-1]
    tgt_idx  = int(snap_idx) - 1 if snap_idx.isdigit() else 0
    fig4 = plot_altitude_profile(result, target_idx=tgt_idx)
    fig4.savefig("nav_altitude.png", dpi=150, facecolor=fig4.get_facecolor())
    print("Saved nav_altitude.png")

    # ── Trajectory GIF ──────────────────────────────────────────────────────
    make_gif(result, "nav_generations.gif", fps=4)
    print("Saved nav_generations.gif")

    plt.show()
