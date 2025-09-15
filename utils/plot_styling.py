from cycler import cycler
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # optional
except Exception:  # pragma: no cover
    sns = None

from config import Config


def apply_dashboard_plot_style() -> None:
    """Apply bright, high-contrast style with pitch-black background and grids.

    Reads configuration from Config.MATPLOTLIB_CONFIG and updates matplotlib rcParams.
    """
    cfg = getattr(Config, 'MATPLOTLIB_CONFIG', {}) or {}

    # Base style and sizes
    plt.style.use(cfg.get('style', 'default'))
    plt.rcParams['figure.figsize'] = cfg.get('figure_size', (12, 8))
    plt.rcParams['font.size'] = cfg.get('font_size', 10)
    plt.rcParams['figure.dpi'] = cfg.get('dpi', 100)
    plt.rcParams['lines.linewidth'] = cfg.get('lines_linewidth', 2.0)

    # Colors and palette
    palette = cfg.get('palette', [])
    if palette:
        plt.rcParams['axes.prop_cycle'] = cycler(color=palette)
        if sns is not None:
            try:
                sns.set_palette(palette)
            except Exception:
                pass

    # Background/foreground (pitch black bg, white fg)
    bg = cfg.get('background_color', '#000000')
    fg = cfg.get('foreground_color', '#FFFFFF')

    plt.rcParams['figure.facecolor'] = bg
    plt.rcParams['axes.facecolor'] = bg
    plt.rcParams['savefig.facecolor'] = bg

    plt.rcParams['text.color'] = fg
    plt.rcParams['axes.labelcolor'] = fg
    plt.rcParams['axes.edgecolor'] = fg
    plt.rcParams['axes.titlecolor'] = fg
    plt.rcParams['xtick.color'] = fg
    plt.rcParams['ytick.color'] = fg

    # Grid settings (enabled globally)
    plt.rcParams['axes.grid'] = cfg.get('axes_grid', True)
    plt.rcParams['grid.alpha'] = cfg.get('grid_alpha', 0.4)
    plt.rcParams['grid.color'] = cfg.get('grid_color', '#333333')
    plt.rcParams['grid.linestyle'] = cfg.get('grid_linestyle', '--')
    plt.rcParams['grid.linewidth'] = 0.8

    # Legend styling for dark background
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.facecolor'] = bg
    plt.rcParams['legend.edgecolor'] = fg
    plt.rcParams['legend.framealpha'] = 0.6

    # Layout convenience
    plt.rcParams['figure.autolayout'] = True 