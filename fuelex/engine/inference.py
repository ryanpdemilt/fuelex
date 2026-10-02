import os
import io
import yaml
import time
import logging
import requests
from retry import retry

import ee

import asyncio
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio as rio

from pyproj import Transformer

import rasterio as rio
from rasterio import Affine as A
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds, bounds

from functools import partial
from multiprocessing.pool import ThreadPool
import threading
from concurrent.futures import ThreadPoolExecutor

from hydra.conf import HydraConf
from omegaconf import OmegaConf, DictConfig
from hydra import initialize,compose
from hydra.utils import instantiate, call

import torch

from .train import preprocessor_factory,model_factory
from fuelex.utils import get_best_model_ckpt_path, get_grid_from_gdf, init_logger

raster_lock = threading.Lock()

AEF_BANDS = [f'A{str(i).zfill(2)}' for i in range(64)]
AEF_CRS='EPSG:4326'
AEF_GSD=10


FM40_LABELS = [
    91, 92, 93, 98, 99,
    101, 102, 103, 104, 105, 106, 107, 108, 109,
    121, 122, 123, 124,
    141, 142, 143, 144, 145, 146, 147, 148, 149,
    161, 162, 163, 164, 165,
    181, 182, 183, 184, 185, 186, 187, 188, 189,
    201, 202, 203, 204,
]

def get_inference_info(group_name,groups,year):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    exp_name = f'{timestamp}_{year}_{group_name}_{'_'.join(groups)}'

    return exp_name

def load_config(exp_dir,logger):
    cfg_path = exp_dir / 'configs' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    logger.info(f'Loading configs from @{cfg_path}')

    return cfg


def load_model(exp_dir,device,logger,cfg):

    final_model_ckpt_path = get_best_model_ckpt_path(exp_dir)

    model_dict = torch.load(final_model_ckpt_path, map_location=device, weights_only=False)
    model = model_factory(cfg)

    model_name = os.path.basename(final_model_ckpt_path).split(".")[0]
    if "model" in model_dict:
        model.load_state_dict(model_dict["model"])
    else:
        model.load_state_dict(model_dict)
    logger.info(f"Loaded {model_name} for inference")

    return model

@retry(tries=25,delay=1,backoff=2)
def pull_single_aef_image(grid_cell,year,dst_crs,dst_scale,band_groups,padding):
    aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filter(ee.Filter.calendarRange(year,year,'year')).mosaic()
    
    # scene_id = grid_cell['sample_id']

    left, bottom, right, top = grid_cell.bounds
    centroid = grid_cell.centroid
    x,y = centroid.x, centroid.y
    pixels = int((right - left) / dst_scale)
    aef_bbox = ee.Geometry.Point((x,y),proj=ee.Projection(dst_crs)).buffer((pixels+padding)*30,proj=ee.Projection(dst_crs))

    transform = A.translation(left - dst_scale / 2,top - dst_scale / 2) * A.scale(dst_scale,-dst_scale)
    
    aef_band_splits = np.array_split(np.array(AEF_BANDS),band_groups)
    partial_results = []
    for i, band_group in enumerate(aef_band_splits):
        # for j in range(spatial_blocks):
        aef_url = aef.getDownloadURL({
            'bands':list(band_group),
            'region':aef_bbox,
            'crs':dst_crs,
            'scale':dst_scale,
            'crsTransform':transform,
            'dimension':[pixels,pixels],
            'format':'NPY'
        })
        # print(f'Download Link Acquired for Scene {scene_id} | Band Group {i}')

        # print(f'Beginning Download for Scene {scene_id} | Band Group {i}')
        r = requests.get(aef_url,stream=True)
        if r.status_code != 200:
            raise r.raise_for_status()
        # print(f'Ending Download for Scene {scene_id} | Band Group {i}')
        
        partial_result = np.load(io.BytesIO(r.content))
        partial_result = np.stack([partial_result[band] for band in band_group])
        partial_results.append(partial_result)
    aef_arr = np.concatenate(partial_results,axis=0)
    
    b, h, w = aef_arr.shape
    if not ((h==pixels) and (w==pixels)):
        center_h = h // 2
        center_w = w // 2

        radius = pixels // 2

        aef_arr = aef_arr[:,center_h-radius:center_h+radius,center_w-radius:center_w+radius]

    return aef_arr

def pull_aef_batch(batch_geometries,year,dst_crs,dst_scale,padding,band_groups,n_jobs):

    batch_items = [row for idx,row in batch_geometries.items()]

    pool = ThreadPool(processes=n_jobs)

    f = partial(pull_single_aef_image,year=year,dst_crs=dst_crs,dst_scale=dst_scale,padding=padding,band_groups=band_groups)

    results = pool.map(f,batch_items)

    aef_batch = np.stack([aef_batch_item for aef_batch_item in results],axis=0)

    return aef_batch

def submit_to_gpu_for_prediction(model,aef_batch,device):
        aef_batch = torch.from_numpy(aef_batch.astype(np.float64)).float().to(device)
        logits = model(aef_batch)
        aef_pred = torch.argmax(torch.softmax(logits,dim=1),dim=1).cpu().numpy()

        return aef_pred

def write_batch_to_rst(aef_pred,batch_geometries,dst_rst_kwargs):
    with rio.open(**dst_rst_kwargs) as dst_rst:
        for i, (idx,row) in enumerate(batch_geometries.items()):
            left,bottom,right,top = row.bounds
            window = from_bounds(
                transform=dst_rst.transform,
                left=left,
                bottom=bottom,
                right=right,
                top=top
            )
            dst_rst.write(aef_pred[i],window=window,indexes=1)
        print('Wrote batch to rst')

def inference_event_loop(model,batch_size,geodf,year,dst_crs,dst_scale,padding,band_groups,device,n_jobs,dst_rst_kwargs):
    idxs = np.arange(len(geodf))

    num_batches = len(idxs) // batch_size
    batch_idxs = np.array_split(idxs,num_batches)
    print(f'Number of Batches to Process: {num_batches}')

    f = partial(
        pull_aef_batch,
        year=year,
        dst_crs=dst_crs,
        dst_scale=dst_scale,
        padding=padding,
        band_groups=band_groups,
        n_jobs=n_jobs
    )

    #retrieve first batch
    current_batch_geometries = geodf.iloc[batch_idxs[0]]
    current_aef_batch = f(current_batch_geometries)

    executor = ThreadPoolExecutor(max_workers=1)
    write_executor = ThreadPoolExecutor(max_workers=1)

    for i, batch in enumerate(batch_idxs[1:]):
        print(f'Procesing Batch {i}')
        
        next_batch_geometries = geodf.iloc[batch]
        future = executor.submit(f,next_batch_geometries)

        current_aef_batch_preds = submit_to_gpu_for_prediction(model,current_aef_batch,device)

        write_executor.submit(
            write_batch_to_rst,
            aef_pred=current_aef_batch_preds,
            batch_geometries=current_batch_geometries,
            dst_rst_kwargs=dst_rst_kwargs
        )
        # write_batch_to_rst(
        #     aef_pred=current_aef_batch_preds,
        #     batch_geometries=current_batch_geometries,
        #     dst_rst=dst_rst
        # )

        current_batch_geometries = next_batch_geometries
        current_aef_batch = future.result()

def log(logger,current_batch,total_batches):
    pass

def run_inference(args):
    exp_dir = Path(args.exp_dir)
    device = args.device
    geometry_path = Path(args.geometry_path)
    out_path = Path(args.out_path)
    group_name = args.group_name
    groups = args.groups
    year = args.year
    scale = args.scale
    crs = args.crs
    batch_size = args.batch_size
    band_groups = args.band_groups
    padding = args.padding
    log_dir = Path(args.log_dir)
    n_classes = args.n_classes

    source = args.source
    if source == 'ee':
        project = args.project
        ee.Authenticate()
        ee.Initialize(
            project=project,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
    n_jobs = args.n_jobs

    exp_name = get_inference_info(group_name,groups,year)

    log_dir = log_dir / f'{exp_name}.log'
    logger = init_logger(log_dir)

    geodf = gpd.read_file(geometry_path)
    print(geodf)
    geodf[group_name] = geodf[group_name].astype(str)
    geodf = geodf.to_crs(crs)
    geodf = geodf[geodf[group_name].isin(groups)]

    left,bottom,right,top = geodf.geometry.union_all().bounds

    dst_transform  = A.translation(left - scale / 2,top - scale / 2) * A.scale(scale,-scale)

    width = (right - left) / scale
    height = (top - bottom) / scale
 

    cfg = load_config(exp_dir,logger)

    img_size = cfg.dataset.img_size
    dropped_labels = cfg.dataset.dropped_labels
    valid_labels = [label for label in FM40_LABELS if not any(label == ignored for ignored in dropped_labels)]
    label_map = dict(zip(valid_labels,np.arange(len(valid_labels))))

    train_transform, test_transform, val_transform = preprocessor_factory(cfg)

    model = load_model(exp_dir,device,logger,cfg)
    model = model.to(device)

    for group in groups:
        group_geo = geodf[geodf[group_name] == group]

        grid = get_grid_from_gdf(group_geo,img_size*scale)
        
        dst_raster_name = out_path / f'{group_name}_{group}_{year}_prediction.tif'
        dst_rst_kwargs = {
            'fp':dst_raster_name,
            'mode':'w',
            'crs':crs,
            'transform':dst_transform,
            'count':1,
            'width':width,
            'height':height,
            'driver':'GTiff',
            'dtype':np.int16,
            'nodata':-9999
        }
    
        dst_rst = rio.open(**dst_rst_kwargs)
        dst_rst.close()
        logger.info(f'Initialized output raster @{dst_rst}')

        inference_event_loop(
            model=model,
            batch_size=batch_size,
            geodf=grid,
            year=year,
            dst_crs=crs,
            dst_scale=scale,
            padding=padding,
            band_groups=band_groups,
            device=device,
            n_jobs=n_jobs,
            dst_rst_kwargs=dst_rst_kwargs
        )




    