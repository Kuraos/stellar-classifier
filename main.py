"""Punto de entrada de stellar-classifier.

Este archivo crea la ventana principal de Tkinter y arranca el bucle de
eventos de la aplicacion.
"""

import tkinter as tk

from gui.app import StellarClassifierApp


def main() -> None:
    """Inicializa la interfaz principal y ejecuta la aplicacion."""
    root = tk.Tk()
    StellarClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
