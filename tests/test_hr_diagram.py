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


def test_plot_hr_highlight_variables_no_error(sample_processed_df) -> None:
    """highlight_variables=True sin columna variable_type no debe fallar."""
    fig = plot_hr(sample_processed_df, highlight_variables=True)
    assert isinstance(fig, Figure)


def test_plot_hr_highlight_variables_with_data(sample_processed_df) -> None:
    """Con variables, el HR debe crear más colecciones visibles."""
    df = sample_processed_df.copy()
    types = ["non_variable"] * len(df)
    types[0] = "DCEP"
    types[1] = "RRAB"
    df["variable_type"] = types
    df["is_variable"] = [tipo != "non_variable" for tipo in types]

    fig_normal = plot_hr(df, highlight_variables=False)
    fig_vars = plot_hr(df, highlight_variables=True)

    ax_normal = fig_normal.axes[0]
    ax_vars = fig_vars.axes[0]
    assert len(ax_vars.collections) >= len(ax_normal.collections)


def test_plot_hr_multiple_calls_same_axes_size(sample_processed_df) -> None:
    """
    Redibujar el HR tres veces sobre el mismo patrón no debe encoger
    el área del axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.hr_diagram import plot_hr

    fig = plt.figure(figsize=(7, 6))

    bboxes = []
    for _ in range(3):
        fig.clf()
        ax = fig.add_subplot(111)
        plot_hr(sample_processed_df, ax=ax)
        fig.canvas.draw()
        bboxes.append(ax.get_position())

    for i in range(1, 3):
        assert abs(bboxes[i].width - bboxes[0].width) < 0.01, (
            f"El axes se encogió en la iteración {i}: "
            f"width={bboxes[i].width:.4f} vs {bboxes[0].width:.4f}"
        )
        assert abs(bboxes[i].height - bboxes[0].height) < 0.01, (
            f"El axes se encogió en la iteración {i}: "
            f"height={bboxes[i].height:.4f} vs {bboxes[0].height:.4f}"
        )

    plt.close(fig)


def test_plot_hr_colorbar_count_stable(sample_processed_df) -> None:
    """
    Tras varios redibujos con limpieza completa, la figura debe mantener
    solo el axes principal y el colorbar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.hr_diagram import plot_hr

    fig = plt.figure(figsize=(7, 6))

    for _ in range(3):
        fig.clf()
        ax = fig.add_subplot(111)
        plot_hr(sample_processed_df, ax=ax)
        fig.canvas.draw()

    assert len(fig.axes) == 2, f"Se esperaban 2 axes (HR + colorbar), hay {len(fig.axes)}"
    plt.close(fig)
