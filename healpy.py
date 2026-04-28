"""Compatibilidad mínima de ``healpy`` para ``dustmaps``.

Este proyecto solo necesita un subconjunto muy pequeño de la API de healpy
para que ``dustmaps`` pueda convertir entre coordenadas angulares y indices
HEALPix. La implementacion usa ``astropy-healpix`` como backend puro en
Python, evitando la dependencia binaria de healpy en este entorno.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy_healpix import HEALPix


def _make_healpix(nside: int, nest: bool) -> HEALPix:
    """Construye un objeto HEALPix con el orden solicitado."""
    if int(nside) <= 0:
        raise ValueError("nside debe ser positivo")
    order = "nested" if nest else "ring"
    return HEALPix(nside=int(nside), order=order, frame="galactic")


def _as_quantity_lonlat(theta, phi, lonlat: bool):
    """Convierte entradas angulares a lon/lat en grados."""
    if lonlat:
        lon = np.asarray(theta, dtype=float) * u.deg
        lat = np.asarray(phi, dtype=float) * u.deg
        return lon, lat

    theta_arr = np.asarray(theta, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    lon = np.degrees(phi_arr) * u.deg
    lat = (90.0 - np.degrees(theta_arr)) * u.deg
    return lon, lat


def ang2pix(nside, theta, phi, nest: bool = False, lonlat: bool = False):
    """Replica mínima de ``healpy.pixelfunc.ang2pix``."""
    hp = _make_healpix(nside, nest)
    lon, lat = _as_quantity_lonlat(theta, phi, lonlat=lonlat)
    return hp.lonlat_to_healpix(lon, lat)


def vec2pix(nside, x, y, z, nest: bool = False):
    """Replica mínima de ``healpy.pixelfunc.vec2pix``."""
    hp = _make_healpix(nside, nest)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    radius = np.sqrt(x_arr * x_arr + y_arr * y_arr + z_arr * z_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        lon = np.degrees(np.arctan2(y_arr, x_arr)) * u.deg
        lat = np.degrees(np.arcsin(np.divide(z_arr, radius, where=radius != 0))) * u.deg
    return hp.lonlat_to_healpix(lon, lat)


def npix2nside(npix):
    """Replica mínima de ``healpy.pixelfunc.npix2nside``."""
    npix_int = int(npix)
    if npix_int <= 0:
        raise ValueError("npix debe ser positivo")
    nside = int(round(np.sqrt(npix_int / 12.0)))
    if 12 * nside * nside != npix_int:
        raise ValueError(f"npix={npix_int} no corresponde a un HEALPix valido")
    return nside


def nside2npix(nside):
    """Replica mínima de ``healpy.pixelfunc.nside2npix``."""
    nside_int = int(nside)
    if nside_int <= 0:
        raise ValueError("nside debe ser positivo")
    return 12 * nside_int * nside_int


pixelfunc = SimpleNamespace(
    ang2pix=ang2pix,
    vec2pix=vec2pix,
    npix2nside=npix2nside,
    nside2npix=nside2npix,
)

__version__ = "compat-astropy-healpix"

__all__ = ["pixelfunc", "ang2pix", "vec2pix", "npix2nside", "nside2npix"]