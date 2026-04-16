from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt

# Academic/publication plotting defaults.
PUB_DPI = 600
RASTER_DPI = 350

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.labelweight": "regular",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply a consistent publication-grade Matplotlib style within a context."""
    with plt.rc_context(PAPER_STYLE):
        yield


def apply_publication_defaults() -> None:
    """Set process-wide plotting defaults for consistent figure styling."""
    plt.rcParams.update(PAPER_STYLE)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = RASTER_DPI) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
