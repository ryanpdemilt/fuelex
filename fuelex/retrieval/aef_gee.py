import ee

import os
import io
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
from multiprocessing.pool import ThreadPool
# mp.set_start_method('thread')
# from concurrent.futures import ThreadPoolExecutor

AEF_BANDS = [f'A{str(i).zfill(2)}' for i in range(64)]
AEF_CRS='EPSG:4326'
AEF_GSD=10

def check_and_make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

@retry(tries=1,delay=1,backoff=2)
def get_gee_chip(geometry,work_dir,year,dst_crs,dst_scale,band_groups,spatial_blocks=None):
    aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filter(ee.Filter.calendarRange(year,year,'year')).mosaic()

    scene_id = geometry['sample_id']
    aef_fname = work_dir / f'AEF_{year}_SCENE{scene_id}.tif'
    left, bottom, right, top = geometry.geometry.bounds
    centroid = geometry.geometry.centroid
    x,y = centroid.x, centroid.y
    t = Transformer.from_crs(crs_from=dst_crs,crs_to=AEF_CRS,always_xy=True)

    aef_left, aef_bottom = t.transform(left,bottom)
    aef_right, aef_top = t.transform(right,top)
    region = ee.Geometry.BBox(aef_left,aef_bottom,aef_right,aef_top)

    pixels = int((right - left) / dst_scale)

    transform = A.translation(left - dst_scale / 2,top - dst_scale / 2) * A.scale(dst_scale,-dst_scale)

    aef_band_splits = np.array_split(np.array(AEF_BANDS),band_groups)

    partial_results = []
    for i, band_group in enumerate(aef_band_splits):
        # for j in range(spatial_blocks):
        aef_url = aef.getDownloadURL({
            'bands':list(band_group),
            'region':region,
            'crs':dst_crs,
            'scale':dst_scale,
            'crsTransform':transform,
            'dimension':[pixels,pixels],
            'format':'NPY'
        })
        print(f'Download Link Acquired for Scene {scene_id} | Band Group {i}')

        print(f'Beginning Download for Scene {scene_id} | Band Group {i}')
        r = requests.get(aef_url,stream=True)
        if r.status_code != 200:
            raise r.raise_for_status()
        print(f'Beginning Download for Scene {scene_id} | Band Group {i}')
        
        
        partial_result = np.load(io.BytesIO(r.content))
        print(f'Retrieved Size {partial_result.shape}')
        partial_result = np.stack([partial_result[band] for band in band_group])
        partial_results.append(partial_result)
    aef_arr = np.concatenate(partial_results,axis=0)
        
    b, h, w = aef_arr.shape
    aef_kwargs = {
        'crs':dst_crs,
        'transform':transform,
        'count':b,
        'width':w,
        'height':h,
        'dtype':np.float32
    }
    print(f'Writing Tif for Scene {scene_id}')
    with rio.open(aef_fname,'w',**aef_kwargs) as rst:
        for i in range(b):
            band_id = i+1
            band = aef_arr[i]
            rst.write(band,band_id)
    


def get_aef_gee(work_dir,grid,project,dst_crs,dst_scale,year,band_groups,n_jobs=1):
    ee.Authenticate()
    ee.Initialize(project=project)

    f = partial(get_gee_chip,work_dir=work_dir,dst_crs=dst_crs,year=year,dst_scale=dst_scale,band_groups=band_groups)

    pool = ThreadPool(processes=n_jobs)

    geoms = [row for (idx,row) in grid.iterrows()]

    pool.map(f,geoms)

    pool.close()


    