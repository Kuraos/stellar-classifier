# stellar-classifier

Pipeline de clasificacion espectral estelar con datos reales de Gaia DR3,
incluyendo procesamiento fisico y una interfaz grafica interactiva en Tkinter.

## Caracteristicas

- Descarga de datos de Gaia DR3 via ADQL con filtros de calidad.
- Conversion de BP-RP a B-V y estimacion de T_eff con relaciones empiricas.
- Calculo de magnitud absoluta, distancia y luminosidad solar.
- Clasificacion espectral OBAFGKM.
- Diagrama Hertzsprung-Russell embebido con herramientas de zoom/pan.
- Tabla ordenable de resultados y exportacion a CSV.

## Captura de la GUI

![GUI placeholder](gui_placeholder.png)

> La imagen es un placeholder. Puedes reemplazarla con una captura real de la app.

## Instalacion

```bash
pip install -r requirements.txt --break-system-packages
```

## Ejecucion

```bash
python main.py
```

## Flujo recomendado

1. Click en **Descargar datos**.
2. Click en **Procesar**.
3. Click en **Graficar**.
4. (Opcional) Click en **Exportar CSV**.

## Estructura principal

```text
stellar-classifier/
├── data/
│   └── download.py
├── src/
│   ├── temperature.py
│   ├── statistics.py
│   ├── hr_diagram.py
│   └── line_fitting.py
├── gui/
│   ├── app.py
│   ├── widgets.py
│   └── plots.py
├── results/
│   └── plots/
├── requirements.txt
└── main.py
```

## Testing

```bash
pytest -q
```

## Referencia de diseno

Ver [DESIGN.md](DESIGN.md) para la especificacion cientifica y arquitectonica completa.