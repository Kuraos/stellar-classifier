# Isócronas PARSEC

Este directorio contiene archivos de isócronas PARSEC en formato CMD 3.7.

## Descarga manual

1. Abre el formulario de Padova en:
   http://stev.oapd.inaf.it/cgi-bin/cmd
2. Selecciona el sistema fotométrico:
   `Gaia DR3 (Vegamags) - DR3`
3. Recomendación de parámetros:
   - `log_age`: desde `7.0` hasta `10.1` en pasos de `0.1`
   - Metalicidad inicial: `[M/H] = 0.0`
4. Exporta el archivo en formato de texto plano CMD 3.7.

## Convención de nombres

Usa un patrón como:

`iso_logage_8.10_mh_0.0.dat`

El nombre debe incluir, de forma legible, el `log_age` y la metalicidad.

## Archivos de ejemplo incluidos

El repositorio incluye archivos sintéticos mínimos para pruebas y para la
demostración de la GUI:

- `iso_logage_7.5_mh_0.0.dat`
- `iso_logage_8.1_mh_0.0.dat`
- `iso_logage_8.8_mh_0.0.dat`
- `iso_logage_9.6_mh_0.0.dat`

Estos archivos respetan la estructura CMD 3.7 y contienen suficientes filas
para que el parser, el filtrado de fases y el trazado funcionen sin depender
de una descarga externa.
