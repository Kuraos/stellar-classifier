# stellar-classifier

Pipeline de clasificacion espectral estelar con datos reales de Gaia DR3,
incluyendo procesamiento fisico y una interfaz grafica interactiva en Tkinter.

## Caracteristicas

- Descarga de datos de Gaia DR3 via ADQL con filtros de calidad.
- Conversion de BP-RP a B-V y estimacion de T_eff con relaciones empiricas.
- Correccion opcional de extincion interestelar con Bayestar19 via dustmaps.
- Precarga Bayestar2019 en segundo plano al iniciar la GUI.
- Calculo de magnitud absoluta, distancia y luminosidad solar.
- Clasificacion espectral OBAFGKM.
- Diagrama Hertzsprung-Russell embebido con herramientas de zoom/pan.
- Tabla ordenable de resultados y exportacion a CSV.
 - Distancias bayesianas (Bailer-Jones et al. 2021) como alternativa a 1000/parallax.

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
3. Si quieres corregir extincion, marca **Corregir extinción** antes de procesar.
	- Si prefieres las distancias bayesianas usa el checkbox "Distancias bayesianas" (independiente de la corrección de extinción).
4. Click en **Graficar**.
5. (Opcional) Click en **Exportar CSV**.
6. La primera vez que abras la GUI, Bayestar2019 se carga en segundo plano.

## Estructura principal

```text
stellar-classifier/
├── data/
│   └── download.py
├── src/
│   ├── temperature.py
│   ├── extinction.py
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