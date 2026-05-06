# stellar-classifier

[![CI](https://github.com/Kuraos/stellar-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Kuraos/stellar-classifier/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-60%25-green)](https://github.com/Kuraos/stellar-classifier/actions)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Pipeline de clasificación espectral estelar con datos reales de **Gaia DR3**.
Deriva parámetros físicos, genera diagramas Hertzsprung–Russell interactivos
y cruza con espectros de LAMOST DR9.

---

## Características

- Derivación de T_eff y clasificación espectral (Ballesteros 2012).
- Corrección de extinción con Bayestar2019 (dustmaps).
- Distancias bayesianas (Bailer-Jones 2021).
- Superposición de isócronas PARSEC y ajuste de edad.
- Cross-match y análisis de espectros LAMOST DR9.

---

## Instalación

### Requisitos del sistema
- Python 3.11 o superior
- Tkinter (Linux: `sudo apt install python3-tk`)
- ~500 MB para mapas de extinción Bayestar2019

### Dependencias

```bash
pip install -r requirements.txt --break-system-packages
```

Para desarrollo (mypy, stubs):

```bash
pip install -r requirements-dev.txt --break-system-packages
```

---

## Uso rápido (GUI)

```bash
python main.py
```

Flujo recomendado en la GUI: Descargar → Procesar → Graficar → Revisar espectros.

---

## Uso programático

Ejemplos rápidos de uso sin lanzar la GUI.

### Descargar muestra Gaia

```python
from data.download import query_gaia_sample

# Descargar 1000 estrellas dentro de 200 pc
df = query_gaia_sample(n_stars=1000, max_dist_pc=200)
print(len(df))
```

### Derivar parámetros físicos

```python
import numpy as np
from src.temperature import bv_from_bprp, teff_from_bv, absolute_magnitude, spectral_type

bprp = df['bp_rp'].to_numpy()
bv = bv_from_bprp(bprp)
teff = teff_from_bv(bv)
df['teff'] = teff
```

### Diagrama HR sin GUI

```python
from src.hr_diagram import plot_hr

fig = plot_hr(df)
fig.savefig('results/hr_diagram.png', dpi=150, bbox_inches='tight')
```

---

## Física implementada

### Temperatura efectiva
Fórmula de Ballesteros (2012):

$$
T_{eff} = 4600\ \left(\frac{1}{0.92(B-V)+1.7} + \frac{1}{0.92(B-V)+0.62}\right)
$$

Transformación BP−RP → B−V por Evans et al. (2018).

### Magnitud absoluta y luminosidad

$$
M_G = m_G + 5 + 5\log_{10}(\pi/1000)
$$

$$
L/L_\odot = 10^{(4.74 - M_G)/2.5}
$$

---

## Referencia de módulos

| Módulo | Función principal | Descripción |
|--------|------------------|-------------|
| `data.download` | `query_gaia_sample` | Descarga TAP Gaia DR3 |
| `src.temperature` | `teff_from_bv` | Ballesteros (2012) |
| `src.extinction` | `apply_extinction_correction` | Bayestar2019 |
| `src.distances` | `best_distance_bayesian` | Bailer-Jones (2021) |
| `src.isochrones` | `load_isochrone` | PARSEC CMD |
| `src.variables` | `add_variability_columns` | Variables Gaia DR3 |
| `src.lamost` | `crossmatch_lamost` | Cross-match LAMOST DR9 |
| `src.hr_diagram` | `plot_hr` | Diagrama HR |

---

## Tests

```bash
# Suite completa (sin red)
pytest -q -m "not online"

# Tests online (requieren red real)
RUN_ONLINE_TESTS=1 pytest -q -m "online"

# Verificación de tipos
mypy src/ data/ main.py
```

---

## Datos externos

- Bayestar2019: descargado por `dustmaps` en `~/.dustmaps/` con `from dustmaps.bayestar import fetch`.
- Isócronas PARSEC: descargar desde http://stev.oapd.inaf.it/cgi-bin/cmd y colocar en `data/isochrones/`.
- Espectros LAMOST DR9: `data/spectra/` (no versionados en repo).

---

## Estructura del proyecto

```text
stellar-classifier/
├── main.py
├── src/
├── data/
├── gui/
├── tests/
└── results/
```

---

## Licencia

GNU General Public License v3.0 — ver [LICENSE](LICENSE).
