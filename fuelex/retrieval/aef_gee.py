import ee

import os
import shutil

import requests
from retry import retry

from tqdm import tqdm
import requests

import numpy as np
import geopandas as gpd

from pyproj import Transformer

import rasterio as rio
from rasterio import Affine as A
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds, bounds

from functools import partial
import multiprocessing as mp

AEF_BANDS = [f'A{str(i).zfill(2)}' for i in range(64)]

def check_and_make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

@retry(tries=10,delay=1,backoff=2)
def get_gee_chip(geometry,work_dir,year,dst_crs,dst_scale,band_groups):
    aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filter(ee.Filter.calendarRange(year,year,'year')).mosaic()

    aef_band_splits = np.array_split(AEF_BANDS,band_groups)

    for band_group in aef_band_splits:
        pass

    


def get_aef_gee(work_dir,grid,project,dst_crs,dst_scale,year,band_groups,n_jobs=1):
    ee.Authenticate()
    ee.Initialize(project=project)

    f = get_gee_chip(work_dir=work_dir,dst_crs=dst_crs,year=year,dst_scale=dst_scale,band_groups=band_groups)

    pool = mp.Pool(processes=n_jobs)

    geoms = [row for (idx,row) in grid.iterrows()]

    pool.map(f,geoms)

    pool.close()


    