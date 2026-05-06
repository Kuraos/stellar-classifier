# stellar-classifier — Documento de Diseño

> Pipeline de clasificación espectral estelar con datos reales de **Gaia DR3** (ESA), con interfaz gráfica interactiva para explorar los resultados.

Este documento es la **especificación completa** del proyecto. Está pensado para que GitHub Copilot (o cualquier asistente de código) pueda generar los archivos necesarios con contexto físico y arquitectónico suficiente.

---

## 1. Objetivo

Construir un pipeline en Python que:

1. Descarga datos fotométricos y astrométricos reales del catálogo **Gaia DR3** vía `astroquery`.
2. Deriva parámetros físicos estelares (T_eff, magnitud absoluta, luminosidad, tipo espectral) mediante relaciones empíricas bien establecidas.
3. Genera un **diagrama Hertzsprung–Russell** automáticamente.
4. Presenta todo en una **interfaz gráfica** (GUI) desde la cual el usuario pueda:
   - Ejecutar cada etapa del pipeline con botones.
   - Visualizar la gráfica HR interactiva (zoom, pan, hover).
   - Ver valores cuantitativos estadísticos en paneles.
   - Inspeccionar una tabla con los datos procesados y ordenables.

---

## 2. Fundamentos físicos

### 2.1 Temperatura efectiva desde el color

La fórmula de **Ballesteros (2012)**, derivada de la ley de Planck de cuerpo negro:

```
T_eff = 4600 · [ 1/(0.92·(B−V) + 1.7)  +  1/(0.92·(B−V) + 0.62) ]   [K]
```

Válida para `0.0 < B−V < 2.0`, con error típico ~1% en secuencia principal.

### 2.2 Conversión Gaia BP−RP → Johnson B−V

Transformación empírica de **Evans et al. (2018)**:

```
B−V = 0.0981 + 0.7119·(BP−RP) + 0.0718·(BP−RP)²
```

### 2.3 Distancia y magnitud absoluta

Gaia entrega el paralaje `π` en miliarcsegundos (mas). Para paralajes de alta calidad (σ_π/π < 0.1):

```
d [pc] = 1000 / π [mas]
M_G    = m_G + 5 + 5·log₁₀(π / 1000)
```

### 2.4 Luminosidad solar
```
L / L_sun = 10^((M_sun − M_G) / 2.5),   M_sun = 4.74
```

### 2.5 Distancias bayesianas (Bailer-Jones)

La aproximación `d = 1000/π` es útil solo como primer orden. Cuando el
paralaje es ruidoso, asimétrico o incluso negativo, la distancia inversa deja
de ser una estimación estadísticamente robusta.

El proyecto usa las distancias precomputadas de Bailer-Jones et al. (2021)
desde `external.gaiaedr3_distance`:

- `r_med_geo`: posterior geométrica con prior espacial.
- `r_med_photogeo`: posterior geométrica + fotometría, preferida cuando existe.
- `r_lo_*` y `r_hi_*`: percentiles 16/84 para incertidumbre asimétrica.

Estrategia de fallback implementada:

1. Usar `r_med_photogeo` si está disponible.
2. Si no, usar `r_med_geo`.
3. Si ambas fallan, volver a `1000/parallax` si el paralaje es positivo.

La GUI permite activar esta ruta sin desactivar la corrección de extinción.
Cuando ambos toggles están activos, la corrección usa `distance_pc_bayesian`.

### 2.6 Isócronas PARSEC

Las isócronas PARSEC (Bressan et al. 2012) representan el lugar geométrico
de estrellas de igual edad y metalicidad en el plano `(log T_eff, M_G)`.
Son una capa visual adicional sobre el HR observacional y se cargan desde
archivos CMD 3.7 exportados manualmente desde Padova.

Uso científico:

- Permiten comparar la secuencia principal observada con modelos teóricos.
- El punto de turnoff ayuda a estimar la edad de cúmulos abiertos.
- La metalicidad por defecto del proyecto es `[M/H] = 0.0`.

Interacción con el flujo del proyecto:

- Las isócronas se dibujan en magnitudes absolutas (`M_G`), por lo que no
   modifican el DataFrame de estrellas observadas.
- La superposición funciona tanto con datos clásicos como con corrección de
   extinción y/o distancias bayesianas.
- El ajuste por χ² explora un grid de edades y devuelve la isócrona que mejor
   se ajusta al subconjunto observado.

### 2.7 Clasificación espectral de Harvard

| Tipo | T_eff (K)        | Color            |
|------|------------------|------------------|
| O    | > 30 000         | Azul intenso     |
| B    | 10 000 – 30 000  | Azul-blanco      |
| A    | 7 500 – 10 000   | Blanco           |
| F    | 6 000 – 7 500    | Blanco-amarillo  |
| G    | 5 200 – 6 000    | Amarillo (Sol)   |
| K    | 3 700 – 5 200    | Naranja          |
| M    | < 3 700          | Rojo             |

### 2.8 Estrellas variables Gaia DR3

El módulo `src/variables.py` normaliza la taxonomía de variabilidad de Gaia DR3 a claves cortas y estables para que la GUI, la tabla y el diagrama HR compartan la misma nomenclatura.

- `VARIABLE_MARKERS` mapea alias y variantes de Gaia a claves canónicas cortas: `DCEP`, `T2CEP`, `RRAB`, `RRC`, `ECL`, `MIRA` y `ROT`.
- `VARIABLE_LABELS` traduce esas claves a etiquetas humanas para el panel lateral y el resumen estadístico.
- `VARIABLE_PLOT_STYLE` define marcadores, colores, tamaños y `zorder` para dibujar cada tipo variable en el HR.
- `PERIOD_LUMINOSITY_TYPES = {"DCEP", "T2CEP", "RRAB", "RRC"}` agrupa los tipos que ya tienen relación período-luminosidad implementada.
- `classify_variable_type(...)` devuelve `non_variable`, `OTHER` o una de las claves cortas anteriores.
- `add_variability_columns(df)` añade `variable_type`, `is_variable`, `pl_period_days` y `distance_pc_PL` sin romper DataFrames que no contengan columnas de variabilidad.
- La distancia P-L usa `phot_g_mean_mag` como aproximación de la magnitud V; queda anotado un TODO para aplicar la transformación G→V de Evans et al. (2018) cuando se requiera precisión publicable.
- En la GUI, el toggle principal y los checkboxes del panel `Estrellas variables` llaman a `_on_variable_filter_changed`, que refresca el HR con `plot_hr(..., highlight_variables=True, variable_types_to_show=...)`.

### 2.9 Espectros LAMOST

LAMOST DR9 aporta millones de espectros publicos de baja resolucion (R~1800)
en el rango aproximado 3700-9000 A. En este proyecto, se integra una ruta
espectroscopica complementaria al pipeline fotometrico de Gaia.

- Cross-match por coordenadas contra Vizier (`V/164`) con radio de 2 arcsec.
- Cache local en `data/spectra/spec_{obsid}.fits` para evitar descargas repetidas.
- Analisis de lineas diagnosticas: H_alpha, H_beta, H_gamma, Ca II K/H, Mg I b y Na I D.
- Medicion de anchos equivalentes via ajuste gaussiano de absorcion.
- Estimacion de tipo espectral Harvard independiente a partir de W_Halpha.
- Estimacion de `T_eff` espectroscopica con relacion empirica simple de Gray (2008):

```
T_eff ~= 456·W + 4180
```

- Validacion cruzada contra `T_eff` fotometrica (Ballesteros 2012).
- La GUI permite click en HR para cargar y mostrar el espectro asociado en la
   pestana nueva de espectroscopia.

Limitacion conocida:

- La comparacion fotometrica-espectroscopica todavia no aplica transformacion
   G->V (TODO futuro con Evans et al. 2018) para escenarios de precision alta.

---

## 3. Arquitectura del proyecto

```
stellar-classifier/
├── README.md
├── DESIGN.md                  ← este archivo
├── requirements.txt
├── .gitignore
├── main.py                    ← lanza la interfaz gráfica
│
├── data/
│   ├── __init__.py
│   └── download.py            ← consulta ADQL a Gaia DR3
│
├── src/
│   ├── __init__.py
│   ├── temperature.py         ← conversiones físicas
│   ├── extinction.py          ← corrección de extinción interestelar
│   ├── hr_diagram.py          ← genera el diagrama HR
│   ├── statistics.py          ← cálculos estadísticos para la GUI
│   └── line_fitting.py        ← (opcional) ajuste de líneas espectrales
│
├── gui/
│   ├── __init__.py
│   ├── app.py                 ← ventana principal (Tkinter)
│   ├── widgets.py             ← componentes reutilizables
│   └── plots.py               ← embebido de matplotlib en Tkinter
│
└── results/
    └── plots/                 ← figuras exportadas
```

---

## 4. Especificación módulo por módulo

### 4.1 `data/download.py`

**Función principal:** `query_gaia_sample(n_stars=5000, max_dist_pc=100) -> pd.DataFrame`

Consulta ADQL a `gaiadr3.gaia_source` con los siguientes filtros de calidad:

- `parallax > 1000/max_dist_pc` (dentro del volumen elegido)
- `parallax_error / parallax < 0.1` (precisión >90%)
- `ruwe < 1.4` (solución astrométrica confiable)
- `phot_bp_rp_excess_factor < 1.5` (fotometría BP/RP válida)

**Columnas requeridas del catálogo:**
`source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, teff_gspphot, lum_flame, radius_flame`

Guarda el resultado como `data/gaia_sample.csv`. Imprime progreso.

### 4.2 `src/temperature.py`

Funciones puras sin dependencias de estado. Todas reciben arrays de NumPy y retornan arrays.

| Función | Entrada | Salida |
|---|---|---|
| `bv_from_bprp(bp_rp)` | array BP−RP | array B−V |
| `teff_from_bv(bv)` | array B−V | array T_eff [K] |
| `absolute_magnitude(g_mag, parallax_mas)` | arrays | array M_G |
| `luminosity_solar(M_abs, M_sun=4.74)` | array | array L/L_sun |
| `spectral_type(teff)` | float o array | str o array de str |

Cada función documentada con docstring incluyendo la referencia bibliográfica.

### 4.2.1 `src/extinction.py`

**Función principal:** `apply_extinction_correction(df, reddening_query=None) -> pd.DataFrame`

Aplica corrección de extinción interestelar sobre una muestra ya descargada
de Gaia. La función:

- Construye coordenadas galacticas con distancia a partir de `ra`, `dec` y `parallax`.
- Consulta Bayestar19 a traves de `dustmaps` para estimar `E(B-V)`.
- Convierte esa señal a `A_V`, `A_G` y `E(BP-RP)` con las relaciones de
   Casagrande et al. (2018).
- Agrega las columnas `A_V`, `A_G`, `E_BR`, `BP_RP_corr`, `B_V_corr`,
   `teff_corr`, `M_G_corr`, `luminosity_solar_corr` y `spectral_type_corr`.
- Recalcula en el mismo DataFrame las columnas canonicas `B_V`, `teff`,
   `M_G`, `luminosity_solar` y `spectral_type` para que la GUI reutilice el
   flujo existente.

La consulta real depende de la disponibilidad del paquete `dustmaps` y del
mapa Bayestar19 local.

### 4.3 `src/statistics.py`

**Función principal:** `compute_statistics(df: pd.DataFrame) -> dict`

Retorna un diccionario con las métricas que la GUI va a mostrar en paneles:

```python
{
    "n_stars": int,
    "teff": {"mean": float, "median": float, "std": float, "min": float, "max": float},
    "M_G": {"mean": float, "median": float, "std": float},
    "distance_pc": {"mean": float, "median": float, "max": float},
    "luminosity_solar": {"mean": float, "median": float, "min": float, "max": float},
    "spectral_distribution": {"O": int, "B": int, "A": int, "F": int, "G": int, "K": int, "M": int},
}
```

### 4.4 `src/variables.py`

**Funciones principales:**

- `classify_variable_type(class_name, period_days=None, classification_result=None) -> str`
- `cepheid_distance(g_mag, period_days, is_type2=False) -> np.ndarray`
- `rrlyrae_distance(g_mag, period_days, metallicity=-1.5) -> np.ndarray`
- `add_variability_columns(df) -> pd.DataFrame`
- `compare_distances(df, distance_col="distance_pc", pl_col="distance_pc_PL") -> dict`

Comportamiento esperado:

- `classify_variable_type` devuelve únicamente claves cortas compatibles con el panel GUI o `non_variable`.
- `add_variability_columns` nunca deja `variable_type` en `None` o `NaN`; las filas sin evidencia quedan marcadas como `non_variable`.
- Los periodos con relación P-L reconocida se elevan a `DCEP`, `T2CEP`, `RRAB` o `RRC` según corresponda.
- `VARIABLE_LABELS` y `VARIABLE_PLOT_STYLE` son los puntos de extensión para la vista de usuario y el marcado visual del HR.
- El código debe degradar silenciosamente cuando falten columnas Gaia de variabilidad.

### 4.5 `src/hr_diagram.py`

**Función principal:** `plot_hr(df, ax=None, use_corrected=False, use_bayesian=False, isochrones_to_overlay=None, highlight_variables=False, variable_types_to_show=None) -> matplotlib.figure.Figure`

- Si `ax` es `None`, crea una figura nueva. Si se pasa un `ax`, dibuja sobre él (clave para embeber en Tkinter).
- Scatter coloreado por T_eff con colormap `RdYlBu_r`.
- Ejes invertidos (convención astronómica: T decreciente a la derecha, M creciente hacia abajo).
- Líneas verticales punteadas marcando los límites de tipos espectrales OBAFGKM.
- Colorbar con T_eff en Kelvin.
- Cuando `highlight_variables=True`, el scatter base se atenúa, las variables se superponen con estilos de `src/variables.py` y la leyenda solo aparece si hay tipos visibles.
- Si faltan columnas de variabilidad, la función degrada silenciosamente al comportamiento normal.

### 4.6 `gui/app.py` — **Interfaz gráfica (requisito central)**

**Stack:** Tkinter (biblioteca estándar, no requiere instalación) + matplotlib embebido.

**Layout de la ventana principal (~1200×800 px):**

```
┌──────────────────────────────────────────────────────────────────────┐
│  stellar-classifier                                    [─] [□] [×]   │
├──────────────────────────────────────────────────────────────────────┤
│  [Descargar datos]  [Procesar]  [Graficar]  [Exportar CSV]           │  ← barra superior
├────────────────────────────┬─────────────────────────────────────────┤
│                            │  Estadísticas                            │
│                            │  ───────────────────                     │
│                            │  N estrellas:     4987                   │
│    [ Diagrama HR ]         │  T_eff media:     5432 K                 │
│    (matplotlib             │  T_eff mediana:   5180 K                 │
│     embebido)              │  Distancia media: 47.3 pc                │
│                            │  Luminosidad media: 0.42 L_sun           │
│                            │                                          │
│                            │  Distribución espectral                  │
│                            │  ───────────────────                     │
│                            │  O: 0    A: 87    G: 512                 │
│                            │  B: 12   F: 354   K: 1823                │
│                            │         M: 2199                          │
├────────────────────────────┴─────────────────────────────────────────┤
│  Tabla de datos (ordenable por columnas, scroll vertical)            │
│  source_id │ ra    │ dec   │ BP-RP │ T_eff │ M_G  │ L/L_sun │ Tipo  │
│  ─────────────────────────────────────────────────────────────────── │
│  1234...   │ 45.2  │ 12.3  │ 0.82  │ 5780  │ 4.83 │ 1.00    │ G     │
│  ...                                                                 │
├──────────────────────────────────────────────────────────────────────┤
│  Estado: listo │ Última actualización: 2026-04-24 14:32              │  ← barra de estado
└──────────────────────────────────────────────────────────────────────┘
```

**Componentes detallados:**

1. **Barra superior de acciones** (`ttk.Frame` con botones):
   - `Descargar datos` → llama a `query_gaia_sample()` en un hilo separado (sin congelar la GUI). Muestra progreso en la barra de estado.
   - `Procesar` → aplica las conversiones de `temperature.py` a la tabla cargada y calcula estadísticas.
   - `Corregir extinción` → activa la corrección de Bayestar19 antes del cálculo de estadisticas y la grafica.
   - `Graficar` → llama a `plot_hr(df, ax=self.ax, highlight_variables=..., variable_types_to_show=...)` sobre el canvas embebido.
   - `Exportar CSV` → guarda el DataFrame procesado en `results/stars_processed.csv`.

La GUI además precarga Bayestar2019 en un hilo de fondo al arrancar. Esto
evita que la primera corrección de extinción tenga que leer el HDF5 grande en
el mismo momento en que el usuario pulsa `Procesar`.

2. **Panel izquierdo — Gráfica** (matplotlib embebido con `FigureCanvasTkAgg`):
   - Toolbar de matplotlib debajo (zoom, pan, guardar imagen).
   - El diagrama HR se redibuja al presionar `Graficar`.

3. **Panel derecho — Estadísticas** (`ttk.LabelFrame`):
   - Dos secciones: "Estadísticas generales" y "Distribución espectral".
   - Valores se actualizan tras `Procesar`.
   - Usar `ttk.Label` con fuente monospace para alineación.

   La GUI también incorpora un panel `Estrellas variables` en ese lateral:

   - El toggle principal activa o desactiva el resaltado de variables en el HR.
   - Los checkboxes por tipo filtran qué clases variables se dibujan.
   - Tras procesar, el panel muestra un resumen con conteos y cuántas estrellas tienen distancia P-L.
   - El cambio de filtros llama a `_on_variable_filter_changed`, que vuelve a dibujar el HR con los tipos activos.

4. **Tabla inferior** (`ttk.Treeview` con `show="headings"`):
   - Columnas ordenables al click en el header.
   - Scroll vertical con `ttk.Scrollbar`.
   - Muestra máximo 500 filas (paginación o truncamiento con aviso).

5. **Barra de estado** (`ttk.Label` en la parte inferior):
   - Muestra el estado actual: "listo", "descargando...", "procesando...", "error: ...".

**Concurrencia:**
- La descarga de Gaia (1–3 min) debe ir en un `threading.Thread` para no bloquear la GUI.
- Usar `self.root.after(100, ...)` para actualizar widgets desde el hilo principal.

**Estado interno de la aplicación** (atributos de la clase `StellarClassifierApp`):
```python
self.df_raw: pd.DataFrame | None        # datos crudos de Gaia
self.df_processed: pd.DataFrame | None  # con columnas T_eff, M_G, L/L_sun, Tipo
self.stats: dict | None                 # output de compute_statistics
self.fig, self.ax                       # matplotlib embebido
```

### 4.7 `main.py`

Archivo minimalista que solo lanza la interfaz:

```python
from gui.app import StellarClassifierApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = StellarClassifierApp(root)
    root.mainloop()
```

---

## 5. Dependencias

`requirements.txt`:

```
astropy>=6.0
astroquery>=0.4.7
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
pandas>=2.0
```

Instalación (sin venv):

```bash
pip install -r requirements.txt --break-system-packages
```

Tkinter viene con Python estándar. En Linux puede requerir: `sudo apt install python3-tk`.

---

## 6. Cómo ejecutar

```bash
python main.py
```

Se abre la ventana. El flujo típico:
1. Click en **Descargar datos** → espera a que termine (ver barra de estado).
2. Click en **Procesar** → se llenan los paneles y la tabla.
3. Click en **Graficar** → aparece el diagrama HR.
4. Opcional: **Exportar CSV** para guardar los datos procesados.

---

## 7. Validación científica

El pipeline debe reproducir las siguientes características esperadas:

- La **secuencia principal** aparece como banda diagonal clara de arriba-izquierda a abajo-derecha.
- Las **gigantes rojas** forman un clump en la zona `log T_eff ∈ [3.6, 3.7], M_G ∈ [−1, 3]`.
- Las **enanas blancas** aparecen en la esquina inferior izquierda (M_G > 10).
- La distribución por tipo espectral refleja la función inicial de masa: mayoría K + M, pocas O + B.
- La correlación entre T_eff calculada y `teff_gspphot` (de Gaia) debe tener R² > 0.85 para FGK.

---

## 8. `.gitignore`

```
__pycache__/
*.pyc
data/*.csv
results/plots/*.png
results/*.csv
.ipynb_checkpoints/
.vscode/
.idea/
```

---

## 9. Referencias

| Referencia | Uso en el proyecto |
|---|---|
| Ballesteros (2012), EPL, 97, 34008 | Fórmula T_eff(B−V) |
| Evans et al. (2018), A&A, 616, A4 | Transformación Gaia → Johnson |
| Gaia Collaboration et al. (2022), A&A, 674, A1 | Catálogo DR3 |
| Bailer-Jones (2021), AJ, 161, 147 | Distancias bayesianas (extensión futura) |
| Schlegel et al. (1998), ApJ, 500, 525 | Corrección de extinción (extensión futura) |

---

## 10. Instrucciones para el asistente de código

**Al generar los archivos:**

1. Respetar **estrictamente** la estructura de carpetas de la sección 3.
2. Cada función de `temperature.py` debe llevar docstring con la referencia correspondiente.
3. El módulo GUI (`gui/app.py`) debe usar **clases** (no código procedural) con una clase principal `StellarClassifierApp`.
4. La descarga de Gaia **debe correr en un hilo** para no congelar la interfaz.
5. El diagrama HR debe dibujarse sobre un `ax` pasado por parámetro, para poder embeberlo en Tkinter.
6. Todos los comentarios y docstrings en **español**.
7. Crear `__init__.py` vacíos en `data/`, `src/` y `gui/` para que los imports funcionen.
8. Generar `requirements.txt` y `.gitignore` como se especifica arriba.
9. El `README.md` del repo debe explicar brevemente el proyecto, mostrar un screenshot (placeholder) de la GUI y listar los pasos de instalación y ejecución.
