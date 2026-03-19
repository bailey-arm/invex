"""
Invex Design Scheme
━━━━━━━━━━━━━━━━━━
Central visual theme for all renders, plots, and UI.
Import from here — never hardcode colors or fonts elsewhere.
"""

# ── Color Palette ──────────────────────────────────────────────────────────────

BACKGROUND = "#0a0e17"         # deep navy-black
SURFACE = "#111827"            # card/panel background
SURFACE_LIGHT = "#1e293b"      # elevated surfaces

GRID = "#1c2940"               # subtle grid lines
GRID_ACCENT = "#263554"        # emphasized grid lines

TEXT_PRIMARY = "#e2e8f0"       # main text
TEXT_SECONDARY = "#64748b"     # labels, captions
TEXT_MUTED = "#334155"         # watermarks, disabled

# ── Accent Colors ──────────────────────────────────────────────────────────────

CYAN = "#06d6a0"               # primary accent — trajectories, highlights
CYAN_GLOW = "#06d6a033"        # glow / translucent overlay
MAGENTA = "#ef476f"            # threats, warnings, enemy paths
MAGENTA_GLOW = "#ef476f33"
ELECTRIC_BLUE = "#118ab2"      # interceptors, friendly assets
ELECTRIC_BLUE_GLOW = "#118ab233"
AMBER = "#ffd166"              # launch points, origins
AMBER_GLOW = "#ffd16633"
WHITE_HOT = "#f8fafc"          # impact points, critical markers

# ── Semantic Mapping ──────────────────────────────────────────────────────────

MISSILE_TRAIL = MAGENTA
MISSILE_TRAIL_GLOW = MAGENTA_GLOW
INTERCEPTOR_TRAIL = ELECTRIC_BLUE
INTERCEPTOR_TRAIL_GLOW = ELECTRIC_BLUE_GLOW
PREDICTED_PATH = CYAN
PREDICTED_PATH_GLOW = CYAN_GLOW
LAUNCH_POINT = AMBER
IMPACT_POINT = WHITE_HOT
TERRAIN_LOW = "#0a0e17"
TERRAIN_HIGH = "#1e3a5f"
TERRAIN_EDGE = "#263554"

# ── Terrain Colormap (low → high elevation) ───────────────────────────────────

TERRAIN_CMAP_STOPS = [
    (0.0,  "#080c14"),
    (0.25, "#0f1b2d"),
    (0.5,  "#1a2d4a"),
    (0.75, "#264060"),
    (1.0,  "#336b87"),
]

# ── Typography ─────────────────────────────────────────────────────────────────

FONT_MONO = "JetBrains Mono"
FONT_SANS = "Inter"
FONT_DISPLAY = "Orbitron"      # titles, HUD elements

FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 9
FONT_SIZE_HUD = 14

# ── Line & Marker Styles ──────────────────────────────────────────────────────

LINE_WIDTH_TRAIL = 1.8
LINE_WIDTH_PREDICTED = 1.2
LINE_WIDTH_GRID = 0.4
MARKER_SIZE = 6
GLOW_ALPHA = 0.2
GLOW_LINE_WIDTH = 6.0          # wide translucent line behind the sharp one

# ── Plot Helpers ───────────────────────────────────────────────────────────────

def apply_theme(ax, title=None):
    """Apply the Invex theme to a matplotlib Axes."""
    fig = ax.get_figure()
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(colors=TEXT_SECONDARY, labelsize=FONT_SIZE_TICK)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_fontsize(FONT_SIZE_LABEL)
    ax.yaxis.label.set_fontsize(FONT_SIZE_LABEL)

    ax.grid(True, color=GRID, linewidth=LINE_WIDTH_GRID, alpha=0.5)

    if title:
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
            pad=16,
        )


def apply_theme_3d(ax, title=None):
    """Apply the Invex theme to a 3D matplotlib Axes."""
    fig = ax.get_figure()
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TEXT_SECONDARY)
        axis.label.set_fontsize(FONT_SIZE_LABEL)
        axis._axinfo["tick"]["color"] = TEXT_SECONDARY
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = LINE_WIDTH_GRID

    ax.tick_params(colors=TEXT_SECONDARY, labelsize=FONT_SIZE_TICK)

    if title:
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
            pad=16,
        )


def trail(ax, x, y, color=MISSILE_TRAIL, glow_color=None, **kwargs):
    """Plot a trajectory trail with a glow effect."""
    glow_color = glow_color or (color + "33")
    ax.plot(x, y, color=glow_color, linewidth=GLOW_LINE_WIDTH, alpha=GLOW_ALPHA, **kwargs)
    ax.plot(x, y, color=color, linewidth=LINE_WIDTH_TRAIL, **kwargs)


def point(ax, x, y, color=LAUNCH_POINT, label=None, **kwargs):
    """Plot a single highlighted point (launch site, impact, etc.)."""
    ax.scatter(x, y, color=color, s=MARKER_SIZE ** 2, zorder=5, label=label, **kwargs)
    ax.scatter(x, y, color=color, s=(MARKER_SIZE * 3) ** 2, alpha=GLOW_ALPHA, zorder=4, **kwargs)


def terrain_cmap():
    """Return a matplotlib LinearSegmentedColormap for terrain rendering."""
    from matplotlib.colors import LinearSegmentedColormap
    stops, colors = zip(*TERRAIN_CMAP_STOPS)
    return LinearSegmentedColormap.from_list("invex_terrain", list(zip(stops, colors)))
