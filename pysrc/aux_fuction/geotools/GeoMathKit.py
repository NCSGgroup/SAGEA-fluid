import calendar
import datetime
import gzip
import math

import numpy as np
import scipy.ndimage as ndi

class GeoMathKit:

    def __init__(self):
        pass

    @staticmethod
    def leakage(ocean_mask, lats, buffer_width_km):
        """
        parameters:
        ocean_mask: 2D numpy (ocean=1, land=0)
        lats: (1D)
        lons: (1D)
        buffer_width_km: km

        return:
        corrected_mask
        """
        if buffer_width_km <=0:
            corrected_mask = ocean_mask
        else:
            # 1. compute grid resolution
            dy = 111.32  # 1 degre of lat ≈ 111.32 km
            dx_at_equator = 111.32  # 1 degree longitude in equator ≈ 111.32 km
            # 2. mean grid resolution
            mean_lat = np.mean(lats)
            dx = dx_at_equator * np.cos(np.deg2rad(mean_lat))
            mean_resolution_km = np.sqrt(dx ** 2 + dy ** 2) / 2  # approximately mean grid resolution
            # 3. the points of buffer
            buffer_cells = int(np.ceil(buffer_width_km / mean_resolution_km))
            # 4. find the boundary
            land = (ocean_mask == 0).astype(int)
            structure = ndi.generate_binary_structure(2, 2)  # 8连通结构(3x3)
            dilated_land = ndi.binary_dilation(land, structure=structure, iterations=buffer_cells)
            # 5. correct mask
            buffer_zone = dilated_land & (ocean_mask == 1)
            corrected_mask = ocean_mask.copy()
            corrected_mask[buffer_zone] = 0
        return corrected_mask

    @staticmethod
    def generate_month_range(start_str, end_str):
        """
        input:2009-01,2010-12
        """
        start = datetime.datetime.strptime(start_str, "%Y-%m")
        end = datetime.datetime.strptime(end_str, "%Y-%m")

        months = []
        current = start
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months
