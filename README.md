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
- Inspeccion de puntos del HR con clic y control visible del modo de grafico.
- Tabla ordenable de resultados y exportacion a CSV.
- Distancias bayesianas (Bailer-Jones et al. 2021) como alternativa a 1000/parallax.
- Isócronas PARSEC (Bressan et al. 2012) para superponer modelos teóricos sobre el HR.
- Estrellas variables Gaia DR3 con marcadores por tipo y validación período-luminosidad.

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
   - Usa el panel **Isócronas PARSEC** para sobreponer modelos o ajustar una edad de referencia.
	- Tras procesar, el panel **Estrellas variables** muestra cuántas variables se detectaron en la muestra.
	- Activa **Mostrar variables en HR** para ver marcadores por tipo.
	- Usa los checkboxes para filtrar por tipo (DCEP, RRAB, ECL, etc.).
	- Pulsa **Validar P-L** para comparar distancias geométricas con la estimación período-luminosidad.
4. Click en **Graficar**.
5. (Opcional) Click en **Exportar CSV**.
6. La primera vez que abras la GUI, Bayestar2019 se carga en segundo plano.
7. En la pestana **Espectroscopia**, pulsa **Buscar espectros LAMOST** para hacer cross-match con LAMOST DR9.
8. Pulsa **Analizar muestra** para descargar y analizar hasta 100 espectros.
9. Haz click sobre un punto del diagrama HR: si tiene espectro disponible, se muestra automaticamente en la pestana de espectroscopia con lineas ajustadas y comparacion de T_eff fotometrica vs espectroscopica.

## Datos externos requeridos

Para usar isócronas PARSEC con archivos reales, descarga los ficheros desde
Padova y colócalos en [data/isochrones/](data/isochrones/). El repositorio
incluye ejemplos sintéticos mínimos para probar la carga y el trazado sin
dependencias externas.

## Mejoras científicas implementadas

- Distancias bayesianas (Bailer-Jones et al. 2021)
- Corrección de extinción (Bayestar2019)
- Isócronas PARSEC (Bressan et al. 2012)
- Estrellas variables Gaia DR3 (Leavitt 1908, Catelan 2009)
- Espectros LAMOST DR9 (Zhao et al. 2012, Gray 2008)

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