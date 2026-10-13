"""Colors shared by the released experiment plots."""

from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

GRAY = "#77727E"

BLUE = "#355070"

ROSE = "#B56576"

PURPLE = "#6D597A"

CORAL = "#E56B6F"

PEACH = "#EAAC8B"

PALETTE = (GRAY, BLUE, ROSE, PURPLE)

EXTENDED_PALETTE = (*PALETTE, CORAL, PEACH)

INK = "#292631"

MUTED = GRAY

EDGE = "#46414F"

GRID = "#D7D3DD"

WHITE = "#FFFFFF"

PROFILE_COLORS = {
    "fixed": GRAY, "narrow_uniform": BLUE,
    "bounded_normal": ROSE, "wide_uniform": PURPLE,
}

HEATMAP_STOPS = ("#F5F6F8", "#D7DCE2", "#AEB9C6", "#72849B", BLUE)

HEATMAP = LinearSegmentedColormap.from_list("paper_muted_tones", HEATMAP_STOPS)

RC_COLORS = {
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.facecolor": WHITE, "figure.facecolor": WHITE, "savefig.facecolor": WHITE,
    "patch.edgecolor": EDGE, "axes.prop_cycle": cycler(color=EXTENDED_PALETTE),
}
