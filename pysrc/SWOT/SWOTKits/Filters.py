# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Filters.py
# @Create  : 11/7/26 22:34
# @Project : SAGEA-fluid
# @Software: PyCharm
"""
Description:
"""

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from scipy.fft import fftfreq, rfftfreq, rfft2, irfft2

def lowpass_2d(data, cutoff_km=6.0, dx_km=2.0,
               dy_km=2.0, order=4, pad=24):
    """
    2D Butterworth low-pass filter.

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        Input grid containing possible NaN values.
    cutoff_km : float
        Cutoff wavelength in kilometers.
    dx_km, dy_km : float
        Grid spacing in kilometers.
    order : int
        Butterworth filter order.
    pad : int
        Number of reflected grid cells at each boundary.

    Returns
    -------
    filtered : ndarray
        Filtered grid, retaining original NaN positions.
    """
    print(f"> > Beginning 2D low pass filter with cutoff {cutoff_km} km.")

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("Input data must be 2D.")

    if cutoff_km <= 0 or dx_km <= 0 or dy_km <= 0:
        raise ValueError("Wavelength and spacing must be positive.")

    if order < 1 or pad < 0:
        raise ValueError("order must be >= 1 and pad >= 0.")

    # 1. Identify valid observations
    valid = np.isfinite(data)

    if not np.any(valid):
        return np.full(data.shape, np.nan)

    # 2. Fill NaN with nearest valid values before FFT
    # FFT cannot directly process NaN values.
    indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True
    )
    filled = data[tuple(indices)]

    # 3. Reflection padding to reduce boundary artifacts
    if pad > 0:
        padded = np.pad(filled, pad, mode="reflect")
    else:
        padded = filled

    ny, nx = padded.shape

    # 4. Spatial frequencies (cycles/km)
    fx = rfftfreq(nx, d=dx_km)
    fy = fftfreq(ny, d=dy_km)

    frequency = np.hypot(fy[:, None], fx[None, :])

    # 5. Butterworth low-pass transfer function
    fc = 1.0 / cutoff_km

    H = 1.0 / (1.0 + (frequency / fc) ** (2 * order))

    # 6. Apply filter in Fourier domain
    F = rfft2(padded)
    filtered_padded = irfft2(F * H, s=padded.shape)

    # 7. Remove padding
    if pad > 0:
        filtered = filtered_padded[pad:-pad, pad:-pad]
    else:
        filtered = filtered_padded

    # 8. Restore original NaN values
    filtered = filtered.copy()
    filtered[~valid] = np.nan
    print(f"> > Finished 2D low pass filter with cutoff {cutoff_km} km.")

    return filtered


# ================================================
# Example: process your existing NetCDF data
# ================================================
def demo():
    with xr.open_dataset("dN_027_584_Along_Track.nc") as ds:
        along = ds["z"].load()

    with xr.open_dataset("dN_027_584_Cross_Track.nc") as ds:
        cross = ds["z"].load()

    # Combine along-track and cross-track components
    data = (along + cross).values

    print("Original shape:", data.shape)
    print("NaN count:", np.isnan(data).sum())

    filtered = lowpass_2d(
        data,
        cutoff_km=6.0,
        dx_km=2.0,
        dy_km=2.0,
        order=4
    )

    print("Filtered shape:", filtered.shape)
    print("NaN count:", np.isnan(filtered).sum())

    # Preserve original grid coordinates
    result = along.copy(data=filtered)
    result.name = "dN_lowpass_6km"
    result.attrs["cutoff_wavelength_km"] = 6.0
    result.attrs["filter"] = "2D Butterworth, order 4"

    # Save filtered result
    result.to_netcdf("dN_027_584_lowpass_6km.nc")
