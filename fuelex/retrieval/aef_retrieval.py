import os
import shutil

import asyncio
import subprocess
import threading

from tqdm import tqdm
import requests

import numpy as np
import geopandas as gpd
from skimage.transform import resize

import networkx as nx

from pyproj import Transformer

import rasterio as rio
from rasterio import Affine as A
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds, bounds

from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# from fuelex.utils import DownloadCache

class AEFManager:

    MANIFEST_URL= 'https://data.source.coop/tge-labs/aef/v1/annual/manifest.txt'
    GEOPARQUET_URL = 'https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet'

    def __init__(
            self,
            work_dir,
            cache_dir,
            cache_size,
            year=2025,
            dst_crs='EPSG:5070',
            dst_scale=30
    ):
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.year = year
        self.dst_crs = dst_crs
        self.dst_scale = dst_scale
  
        # self.cache = DownloadCache(cache_size=5)

    def check_and_get_manifest(self):
        manifest_file = os.path.join(self.work_dir,'manifest.txt')

        if not os.path.exists(manifest_file):
            r = requests.get(self.MANIFEST_URL)
            chunk_size=512*1024
            with open(manifest_file,'wb') as f:
                total_length = r.headers.get('content-length')
                
                if total_length is None:
                    f.write(r.content)
                else:
                    total_length = int(total_length)
                    bar = tqdm(r.iter_content(chunk_size=chunk_size),desc=self.MANIFEST_URL,total=int(total_length // chunk_size),unit='iB',unit_scale=True,unit_divisor=chunk_size)
                    
                    for chunk in bar:
                        if chunk:
                            f.write(chunk)
                            f.flush()
        else:
            print(f'Manifest found @ {manifest_file}')

    def check_and_get_geoparquet(self):

        parquet_file = os.path.join(self.cache_dir,'aef_index_stac_geoparquet.parquet')

        if not os.path.exists(parquet_file):
            r = requests.get(self.GEOPARQUET_URL)
            chunk_size=512*1024
            with open(parquet_file,'wb') as f:
                total_length = r.headers.get('content-length')
                
                if total_length is None:
                    f.write(r.content)
                else:
                    total_length = int(total_length)
                    bar = tqdm(r.iter_content(chunk_size=chunk_size),desc=self.GEOPARQUET_URL,total=int(total_length // chunk_size),unit='iB',unit_scale=True,unit_divisor=chunk_size)
                    
                    for chunk in bar:
                        if chunk:
                            f.write(chunk)
                            f.flush()
        else:
            print(f'Parquet Index Found @ {parquet_file}')
        aef_parquet_index = gpd.read_parquet(parquet_file)
        return aef_parquet_index

    def pull_aef_sample(self,sample_id,grid_geom,grid_tile,padding=10):
        dst_crs = self.dst_crs
        dst_bounds = grid_geom.geometry.bounds
        dst_left, dst_bottom, dst_right, dst_top = dst_bounds

        outfile_name = f'AEF_{self.year}_Scene{sample_id}.tif'

        dst_transform = A.translation(dst_left - self.dst_scale / 2,dst_top - self.dst_scale / 2) * A.scale(self.dst_scale,-self.dst_scale) 

        # If the sample is contained in a single AEF tile
        if len(grid_tile) == 1:

            aef_fname = grid_tile.iloc[0]['assets']['data']['href'].split('/')[-1]

            with rio.open(self.cache_dir / aef_fname) as src_rst:
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
                    resolution=self.dst_scale,
                    left=src_left,
                    bottom=src_bottom,
                    right=src_right,
                    top=src_top
                )
                with WarpedVRT(src_rst,src_crs=src_rst.crs,transform=warp_transform,crs=dst_crs,height=warp_height,width=warp_width,resampling=Resampling.bilinear) as src_warped_rst:
                    left, bottom, right, top = dst_bounds

                    window = from_bounds(
                        left,
                        bottom,
                        right,
                        top,
                        src_warped_rst.transform
                    )

                    sample = src_warped_rst.read(window=window)
        # If we need to combine windows from multiple AEF tiles to construct our sample
        else:

            dst_bounds = grid_geom.geometry.bounds
            dst_left,dst_bottom,dst_right,dst_top = dst_bounds
            final_shape_pixels = int((dst_right - dst_left)  // self.dst_scale)

            sample = np.full((64,final_shape_pixels,final_shape_pixels),-128)


            for idx, tile in grid_tile.iterrows():
                aef_fname = tile['assets']['data']['href'].split('/')[-1]

                with rio.open(self.cache_dir / aef_fname) as src_rst:
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
                        resolution=self.dst_scale,
                        left=src_left,
                        bottom=src_bottom,
                        right=src_right,
                        top=src_top
                    )
                    with WarpedVRT(src_rst,transform=warp_transform,crs=dst_crs,height=warp_height,width=warp_width) as src_warped_rst:
                        warped_bounds = src_warped_rst.bounds
                        left, bottom, right, top = dst_bounds

                        full_rst_window = Window(0,0,src_warped_rst.width,src_warped_rst.height)
    
                        window = from_bounds(
                            left,
                            bottom,
                            right,
                            top,
                            src_warped_rst.transform
                        )

                        try: 
                            intersecting_window = window.intersection(full_rst_window)
                            window_left,window_bottom,window_right,window_top = bounds(intersecting_window,src_warped_rst.transform)

                            sample_window =  from_bounds(
                                window_left,
                                window_bottom,
                                window_right,
                                window_top,
                                dst_transform
                            )

                            col_off,row_off = int(sample_window.col_off),int(sample_window.row_off)
                            
                            windowed_sample = src_warped_rst.read(window=intersecting_window)
                            # print(windowed_sample)
                            _, window_h, window_w = windowed_sample.shape
                            print(f'col_off {col_off} | row_off {row_off} | width {window_w} | height {window_h}')
                            print(f'Window Sample Shape: {windowed_sample.shape}')

                            #preserve all true negatives but if any raster has valid values prefer those
                            

                            subsample = sample[:,row_off:row_off+window_h,col_off:col_off+window_w]
                            #correct for small errors in transformation sizing
                            if subsample.shape != windowed_sample.shape:
                                windowed_sample = resize(np.transpose(windowed_sample,(1,2,0)),(subsample.shape[1],subsample.shape[2],subsample.shape[0]))
                                windowed_sample = np.transpose(windowed_sample,(2,0,1))

                            mask = windowed_sample == -128

                            subsample = np.where(mask,subsample,windowed_sample)
                            sample[:,row_off:row_off+window_h,col_off:col_off+window_w] = subsample
                        except:
                            print('Empty window intersection found, subset not assigned')
                        # print(intersecting_window.width,intersecting_window.height)



        c, h, w = sample.shape
        dst_kwargs = {
            'crs':dst_crs,
            'width':w,
            'height':h,
            'count':c,
            'transform':dst_transform,
            'driver':'GTiff',
            'dtype':'int16',
            'nodata':-128
        } 
        with rio.open(self.work_dir / outfile_name,'w',**dst_kwargs) as dst_rst:
            for i in range(c):
                dst_rst.write(sample[i],i+1)    


    def cache_aef_tile(self,href):
        dl_link = href['assets']['data']['href'].split('/')
        dl_link.pop(2)
        dl_link = '/'.join(dl_link)
        
        if (self.cache_dir / dl_link.split('/')[-1]).exists():
            print(f'Found cached file')
        else:
            print(f'Initiating download for {dl_link}')
            subprocess.run(['aws','s3','cp', dl_link, self.cache_dir, '--endpoint-url', 'https://data.source.coop','--no-sign-request'])
            print(f'Completing download for {dl_link}')

    def clean_cache(self):
        tif_files = self.cache_dir.glob('*.tiff')

        for file in tif_files:
            try:
                os.remove(file)
            except:
                print(f'Unable to clear {file} from cache')

    # async def producer(self,dependency_group,overlapping_aef_tiles):
    #     pass
    # async def consumer(self,queue):
    #     while True:
    #         pass

    def get_aef_samples(self,grid,n_jobs=1,buffer=300,cleanup=True):
        aef_index = self.check_and_get_geoparquet()
        aef_index = aef_index[aef_index.datetime.dt.year == self.year]

        overlapping_aef_tiles = grid.to_crs(aef_index.crs).sjoin(aef_index[['id','assets','geometry','proj:epsg']],predicate='intersects')
        overlapping_aef_tiles = overlapping_aef_tiles.to_crs(grid.crs)

        sample_ids = overlapping_aef_tiles['sample_id'].unique()
        unique_tiles = overlapping_aef_tiles['id'].unique()
        # required_tile_combinations =  overlapping_aef_tiles['sample_id'].map(lambda x: overlapping_aef_tiles[overlapping_aef_tiles['sample_id'] == x]['id'].values())

        n_tiles = len(unique_tiles)
        dependency_list = grid['sample_id'].map(lambda x:overlapping_aef_tiles[overlapping_aef_tiles['sample_id'] == x]['id'].values)
        dependency_list = [[str(element)for element in arr] for arr in dependency_list.values]
        edge_list = self.build_dependency_graph(dependency_list)
        dependency_graph = nx.Graph(edge_list)
        dependency_graph.remove_edges_from(list(nx.selfloop_edges(dependency_graph)))
        ccs = nx.connected_components(dependency_graph)
        

        groups = []
        cc = next(ccs)
        current_group = [cc]
        group_size = len(cc)

        for cc in ccs:
            print(cc)
            new_elements = len(cc)
            if (group_size + new_elements) > self.cache_size:
                groups.append(current_group)
                current_group = [cc]
                group_size = new_elements
            else:            
                group_size = group_size + len(cc)
                current_group.append(cc)
        groups.append(current_group)

        for group in groups:
            for cc in group:
                print(cc)
                group_aef_tiles = overlapping_aef_tiles[overlapping_aef_tiles['id'].isin(cc)]
                for idx, row in group_aef_tiles.iterrows():
                    self.cache_aef_tile(row)

                pool = mp.Pool(processes=n_jobs)

                combinations = []
                group_sample_ids = group_aef_tiles['sample_id'].unique()
                for sample_id in group_sample_ids:
                    sample_geometry = group_aef_tiles[group_aef_tiles.sample_id == sample_id].iloc[0]              
                    sample_aef_tiles = group_aef_tiles[group_aef_tiles.sample_id == sample_id]

                    element = (sample_id,sample_geometry,sample_aef_tiles)
                    combinations.append(element)

                pool.starmap(self.pull_aef_sample,combinations)
                pool.close()

            if cleanup:
                self.clean_cache()

    def build_dependency_graph(self,dependency_list):
        nodes = set(str(element) for arr in dependency_list for element in arr)
        edges  = dict()

        edge_list = {}
        for node in nodes:
            edges[node] = set([node])
            for dependency in dependency_list:
                if (node in dependency) and (len(dependency) != 1):
                    edges[node] = edges[node] | set(dependency)

        edge_list = [(node,edge) for node in nodes for edge in edges[node]]

        return edge_list