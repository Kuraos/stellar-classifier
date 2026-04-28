from __future__ import annotations

from matplotlib.figure import Figure

from src.hr_diagram import plot_hr


def test_plot_hr_returns_figure(sample_processed_df) -> None:
    fig = plot_hr(sample_processed_df)
    assert isinstance(fig, Figure)


def test_plot_hr_inverts_axes_and_adds_colorbar(sample_processed_df) -> None:
    fig = plot_hr(sample_processed_df)
    ax = fig.axes[0]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert xlim[0] > xlim[1]
    assert ylim[0] > ylim[1]
    assert len(fig.axes) >= 2


def test_plot_hr_reuses_single_colorbar_on_replot(sample_processed_df) -> None:
    fig = plot_hr(sample_processed_df)
    ax = fig.axes[0]

    plot_hr(sample_processed_df, ax=ax)

    assert len(fig.axes) == 2
