import os
import io
import yaml
import time
import logging
import requests
from retry import retry

import asyncio
import subprocess
from pathlib import Path

from tqdm import tqdm

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
from fuelex.utils import get_best_model_ckpt_path, get_grid_from_gdf, init_logger,dequantize


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
GEOPARQUET_URL = 'https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet'

def check_and_get_geoparquet(cache_dir):

    parquet_file = os.path.join(cache_dir,'aef_index_stac_geoparquet.parquet')

    if not os.path.exists(parquet_file):
        r = requests.get(GEOPARQUET_URL)
        chunk_size=512*1024
        with open(parquet_file,'wb') as f:
            total_length = r.headers.get('content-length')
            
            if total_length is None:
                f.write(r.content)
            else:
                total_length = int(total_length)
                bar = tqdm(r.iter_content(chunk_size=chunk_size),desc=GEOPARQUET_URL,total=int(total_length // chunk_size),unit='iB',unit_scale=True,unit_divisor=chunk_size)
                
                for chunk in bar:
                    if chunk:
                        f.write(chunk)
                        f.flush()
    else:
        print(f'Parquet Index Found @ {parquet_file}')
    aef_parquet_index = gpd.read_parquet(parquet_file)
    return aef_parquet_index

def cache_aef_tile(cache_dir,href,dst_crs=None,dst_scale=None):
    dl_link = href['assets']['data']['href'].split('/')
    output_file_name = dl_link[-1]
    dl_link.pop(2)
    dl_link = '/'.join(dl_link)
    
    if (cache_dir / output_file_name).exists():
        print(f'Found cached file')
    else:
        print(f'Initiating download for {dl_link}')
        subprocess.run(['aws','s3','cp', dl_link, cache_dir, '--endpoint-url', 'https://data.source.coop','--no-sign-request'])
        print(f'Completing download for {dl_link}')

    if dst_crs and dst_scale:
        with rio.open(cache_dir / output_file_name) as src_rst:
            src_transform = src_rst.transform
            src_crs = src_rst.crs
            src_bounds = src_rst.bounds
            src_left,src_bottom,src_right,src_top = src_rst.bounds
            width, height = src_rst.width, src_rst.height

            warp_transform, warp_width, warp_height = calculate_default_transform(
                src_crs=src_crs,
                dst_crs=dst_crs,
                width=width,
                height=height,
                resolution=dst_scale,
                left=src_left,
                bottom=src_bottom,
                right=src_right,
                top=src_top
            )
            kwargs = src_rst.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': warp_transform,
                'width': warp_width,
                'height': warp_height
            })
            with rio.open(cache_dir / f"reprojected_{output_file_name}","w",**kwargs) as dst_rst:
                for i in range(1, src_rst.count + 1):
                    reproject(
                        source=rio.band(src_rst, i),
                        destination=rio.band(dst_rst, i),
                        src_transform=src_rst.transform,
                        src_crs=src_rst.crs,
                        dst_transform=warp_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )


def clean_cache(cache_dir):
    tif_files = cache_dir.glob('*.tiff')

    for file in tif_files:
        try:
            os.remove(file)
        except:
            print(f'Unable to clear {file} from cache')


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

def load_chip(aef_tile,grid_tile,cache_dir,dst_crs,dst_scale):
    dst_bounds = grid_tile.geometry.bounds
    dst_left, dst_bottom, dst_right, dst_top = dst_bounds
    aef_fname = f'reprojected_{aef_tile['assets']['data']['href'].split('/')[-1]}'

    with rio.open(cache_dir / aef_fname) as src_rst:
        src_transform = src_rst.transform

        window = from_bounds(
            dst_left,
            dst_bottom,
            dst_right,
            dst_top,
            src_transform
        )

        sample = src_rst.read(window=window,boundless=True,fill_value=0)
        sample = dequantize(sample)

    return sample

def load_batch(grid_tiles,aef_tile,cache_dir,dst_crs,dst_scale,device):
    print(grid_tiles)
    aef_batch = []
    for idx,grid_tile in grid_tiles.iterrows():
        print(f'Loading item {idx} in batch')
        sample=load_chip(
            grid_tile=grid_tile,
            aef_tile=aef_tile,
            cache_dir=cache_dir,
            dst_crs=dst_crs,
            dst_scale=dst_scale
        )
        print(f'Sample {idx} loaded w/ shape: {sample.shape}')
        aef_batch.append(sample)
    aef_batch = np.stack(aef_batch,axis=0)
    aef_batch = torch.from_numpy(aef_batch.astype(np.float64)).float().to(device)
    
    return aef_batch

def tile_inference(
        model,
        batch_size,
        device,
        grid,
        aef_tile,
        cache_dir,
        dst_crs,
        dst_scale,
        dst_rst_kwargs,
        write_out_executor
    ):
    idxs = np.arange(len(grid))

    print(f'Number of Grid Cells {len(grid)} | Batch Size {batch_size}')
    num_batches = len(idxs) // batch_size
    if (len(idxs) % batch_size) != 0:
        num_batches = num_batches + 1
    batch_idxs = np.array_split(idxs,num_batches)

    print(f'Batch Size: {batch_size} | num_batches {num_batches}')

    data_loader_executor = ThreadPoolExecutor(max_workers=4)

    b = partial(
        load_batch,
        aef_tile=aef_tile,
        cache_dir=cache_dir,
        dst_crs=dst_crs,
        dst_scale=dst_scale,
        device=device
    )

    batch_futures = [data_loader_executor.submit(b,grid.iloc[batch]) for batch in batch_idxs]
    write_out_futures = []

    for i, (batch,batch_future) in enumerate(zip(batch_idxs,batch_futures)):
        batch_geometries = grid.iloc[batch]
        aef_batch = batch_future.result()

        aef_preds = submit_to_gpu_for_prediction(model,aef_batch)
        print(f'Predicted batch {i} w/ shape {aef_preds.shape}')

        write_batch_to_rst(
            aef_preds,
            batch_geometries,
            dst_rst_kwargs
        )

        # write_out_futures.append(
        #     write_out_executor.submit(
        #         write_batch_to_rst,
        #         aef_preds,
        #         batch_geometries,
        #         dst_rst_kwargs
        #     )
        # )
    # write_out_results = [write_out_future.result() for write_out_future in write_out_futures]


def submit_to_gpu_for_prediction(model,aef_batch):
    logits = model(aef_batch)
    aef_pred = torch.argmax(torch.softmax(logits,dim=1),dim=1).cpu().numpy()
    return aef_pred

def write_batch_to_rst(aef_pred,batch_geometries,dst_rst_kwargs):

    with rio.open(**dst_rst_kwargs) as dst_rst:
        for i, (idx,row) in enumerate(batch_geometries.iterrows()):
            left,bottom,right,top = row.geometry.bounds

            print(left,bottom,right,top)
            window = from_bounds(
                transform=dst_rst.transform,
                left=left,
                bottom=bottom,
                right=right,
                top=top
            )

            intersecting_window = window.intersection(Window(0,0,dst_rst.width,dst_rst.height))
            interesetion_width = int(intersecting_window.width)
            intersection_height = int(intersecting_window.height)

            window_col_off, window_row_off = int(window.col_off), int(window.row_off)
            if (window_col_off < 0) or (window_row_off < 0):
                sample = aef_pred[i,window_col_off:,window_row_off:]
            else:
                sample = aef_pred[i,:intersection_height,:interesetion_width]

            try:
                dst_rst.write(sample,window=intersecting_window,indexes=1)
            except:
                print('Unsuccesful write, ignoring sample')
        print('Wrote batch to rst')


def aef_ineference_event_loop(
        model,
        batch_size,
        device,
        overlapping_aef_tiles,
        cache_dir,
        cache_size,
        dst_crs,
        dst_scale,
        dst_rst_kwargs,
        cleanup
    ):

    cache_fill_executor = ThreadPoolExecutor(max_workers=1)
    inference_executor = ThreadPoolExecutor(max_workers=1)
    write_executor = ThreadPoolExecutor(max_workers=1)

    aef_tiles_ids = overlapping_aef_tiles['id'].unique()

    n_groups = len(aef_tiles_ids) // cache_size
    aef_groups = np.array_split(aef_tiles_ids,n_groups)
    for group in aef_groups:
        print(f'Print processing cache w/ group members {group}')
        inference_futures = []
        if len(group) > 1:
            current_href = overlapping_aef_tiles[overlapping_aef_tiles['id'] == group[0]].iloc[0]
            future = cache_fill_executor.submit(
                cache_aef_tile,
                cache_dir=cache_dir,
                href=current_href,
                dst_crs=dst_crs,
                dst_scale=dst_scale
            )
            current_tile_grid = overlapping_aef_tiles[overlapping_aef_tiles['id'] == group[0]]

            for tile_id in group[1:]:
                #download_completed
                future.result()

                #submit next inference/tif write
                next_href = overlapping_aef_tiles[overlapping_aef_tiles['id'] == tile_id].iloc[0]
                next_tile_grid = overlapping_aef_tiles[overlapping_aef_tiles['id'] == tile_id]
                future = cache_fill_executor.submit(
                    cache_aef_tile,
                    cache_dir=cache_dir,
                    href=next_href,
                    dst_crs=dst_crs,
                    dst_scale=dst_scale
                )
                #submit inference computation

                tile_inference(
                    model,
                    batch_size,
                    device,
                    current_tile_grid,
                    current_href,
                    cache_dir,
                    dst_crs,
                    dst_scale,
                    dst_rst_kwargs,
                    write_executor
                )
                # inference_futures.append(
                #     inference_executor.submit(
                #         tile_inference,
                #         model,
                #         batch_size,
                #         device,
                #         current_tile_grid,
                #         current_href,
                #         cache_dir,
                #         dst_crs,
                #         dst_scale,
                #         dst_rst_kwargs,
                #         write_executor
                #     )
                # )
                current_href = next_href
                current_tile_grid = next_tile_grid

            #wait for tile inference and writes to finish
            inference_results = [inference_future.result() for inference_future in inference_futures]
            # write_results = [write_future.result() for write_future in write_out_futures]
        else:
            current_tile_grid = overlapping_aef_tiles[overlapping_aef_tiles['id'] == group[0]]
            current_href = overlapping_aef_tiles[overlapping_aef_tiles['id'] == group[0]].iloc[0]
            cache_aef_tile(
                cache_dir=cache_dir,
                href=current_href,
                dst_crs=dst_crs,
                dst_scale=dst_scale
            )
            tile_inference(
                model,
                batch_size,
                device,
                current_tile_grid,
                current_href,
                cache_dir,
                dst_crs,
                dst_scale,
                dst_rst_kwargs,
                write_executor
            )
        # clean cache when writes are done    
        if cleanup:
            clean_cache(cache_dir)

def run_inference_source_coop(args):
    exp_dir = Path(args.exp_dir)
    device = args.device
    geometry_path = Path(args.geometry_path)
    out_path = Path(args.out_path)
    cache_dir = Path(args.cache_dir)
    cache_size = args.cache_size
    group_name = args.group_name
    groups = args.groups
    year = args.year
    scale = args.scale
    crs = args.crs
    batch_size = args.batch_size
    padding = args.padding
    log_dir = Path(args.log_dir)
    cleanup = args.cleanup
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

    aef_index = check_and_get_geoparquet(cache_dir)
    aef_index = aef_index[aef_index.datetime.dt.year == year]

    for group in groups:
        group_geo = geodf[geodf[group_name] == group]

        grid = get_grid_from_gdf(group_geo,img_size*scale)
        grid = gpd.GeoDataFrame(geometry=grid,crs=grid.crs)
        
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
        dst_rst_kwargs.update({'mode':'r+'})


        overlapping_aef_tiles = grid.to_crs(aef_index.crs).sjoin(aef_index[['id','assets','geometry','proj:epsg']],predicate='intersects')
        overlapping_aef_tiles = overlapping_aef_tiles.to_crs(grid.crs)

        aef_ineference_event_loop(
            model=model,
            batch_size=batch_size,
            device=device,
            overlapping_aef_tiles=overlapping_aef_tiles,
            cache_dir=cache_dir,
            cache_size=cache_size,
            dst_crs=crs,
            dst_scale=scale,
            dst_rst_kwargs=dst_rst_kwargs,
            cleanup=cleanup
        )