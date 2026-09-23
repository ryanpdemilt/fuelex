import pathlib
from pathlib import Path

import numpy as np
import geopandas as gpd

import rasterio as rio
from rasterio import Affine as A
from rasterio.windows import Window,from_bounds

import multiprocessing as mp

from fuelex.retrieval.sample import sample_images_from_geo_by_key

SUPERZONES = {
    'Northwest':[1,2,7,8,9],
    'Pacific South':[3,4,5,6],
    'Great Basin': [12,13,16,17,18],
    'Northern Rockies':[10,19,20,21,22],
    'Southwest':[14,15,23,24,28],
    'Northern Plains':[29,30,31,33,38,39,40,42,43],
    'South Central West':[25,26,27,32,34,35,36],
    'South Central East':[37,44,45,98],
    'Great Lakes':[41,49,50,51,52,62],
    'Southern Appalachians':[47,48,53,54,57,59],
    'Southeast':[46,55,56,58,99],
    'Northeast':[60,61,63,64,65,66]
}
zone_to_superzone_map = dict([(v_prime,k) for (k,v) in SUPERZONES.items() for v_prime in v])

def map_superzone(zone):
    return zone_to_superzone_map[zone]

def sample_from_landfire_geometry(expname,geo_file,group_names,n_samples_per_group,img_size,out_file,seed):
    landfire_gsd = 30
    geodf = gpd.read_file(geo_file)
    geodf['superzone'] = geodf['ZONE_NUM'].map(lambda x: zone_to_superzone_map[x])
    sample_output_file = sample_images_from_geo_by_key(expname,geodf,group_names,img_size,landfire_gsd,n_samples_per_group,out_file,seed)


def sample_landfire_ims(work_dir,src_path,dataset,year,geometry,dst_scale=30,n_jobs=1):

    with rio.open(src_path / f"LF{year}_{dataset.upper()}_CONUS.tif") as src_rst:

        for idx, sample in geometry.iterrows():
            outfilename = f'{dataset.upper()}_{year}_Scene{sample['sample_id']}.tif'
            geo = sample.geometry
            left, bottom, right,top = geo.bounds
            window = from_bounds(
                transform=src_rst.transform,
                left=left,
                bottom=bottom,
                right=right,
                top=top
            )

            sample = src_rst.read(1,window=window)
            h,w = sample.shape

            dst_transform = A.translation(left - dst_scale /2, top - dst_scale / 2) * A.scale(dst_scale,-dst_scale)
            dst_kwargs = {
                'width':w,
                'height':h,
                'count':1,
                'crs':geometry.crs,
                'transform':dst_transform,
                'dtype':'uint16',
                'driver':'GTiff'
            }

            with rio.open(work_dir / outfilename,'w+',**dst_kwargs) as dst_rst:
                dst_rst.write(sample,1)
                
