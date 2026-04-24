"""Widgets de graficacion matplotlib embebidos en Tkinter."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class MatplotlibPanel(ttk.Frame):
    """Panel reutilizable para embebido de matplotlib en Tkinter."""

    def __init__(self, master: tk.Misc, width: float = 6.5, height: float = 5.5, dpi: int = 100):
        super().__init__(master)

        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.clear(message="Presiona 'Graficar' para mostrar el diagrama HR")

    def clear(self, message: str = "") -> None:
        """Limpia el eje actual y opcionalmente muestra un mensaje central."""
        self.ax.clear()
        if message:
            self.ax.text(
                0.5,
                0.5,
                message,
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
            )
            self.ax.set_axis_off()
        self.canvas.draw_idle()

    def draw(self) -> None:
        """Fuerza repintado del canvas."""
        self.canvas.draw_idle()
