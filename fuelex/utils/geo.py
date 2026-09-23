import torch

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.ops import unary_union

def grid_bounds(geom, delta,correct_egdes=False):
    minx, miny, maxx, maxy = geom.bounds

    delx = (maxx - minx) % delta
    dely = (maxy - miny) % delta

    if (delx < 1e-6) and (dely < 1e-6):
        nx = int((maxx - minx)/delta)
        ny = int((maxy - miny)/delta)
        gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
    else:
        nx = int((maxx - minx)/delta) + 2
        ny = int((maxy - miny)/delta) + 2
        maxx = minx + (nx * delta)
        maxy = miny + (ny * delta)

        gx = np.array([minx + (i * delta) for i in range(nx)])
        gy = np.array([miny + (i * delta) for i in range(ny)])

    # gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)

    grid = []
    for i in range(len(gx)-1):
        for j in range(len(gy)-1):
            poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
            grid.append( poly_ij )
    return grid

def partition(geom, delta):
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, delta)))
    return grid

def union(polys):
    merged = unary_union(polys)

    return gpd.GeoSeries([merged])