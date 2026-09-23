import os
import sys

import pathlib
from pathlib import Path

import pandas as pd
import geopandas as gpd

from rich import print

from fuelex.utils import partition, union

def sample_images_from_geo_by_key(dataset_name,geodf,key_name,spatial_size_pixels,spatial_gsd,n_sample_ims,out_file,splits=None,split_sizes=None,seed=2000):
    spatial_size = spatial_size_pixels*spatial_gsd
    # try:
    #     if isinstance(geo_file,str) and geo_file.split('.') == 'parquet':
    #         geodf = gpd.read_parquet(geo_file)
    #     else:
    #         geodf = gpd.read_file(geo_file)
        
    # except:
    #     print(f'[red]Error[\red]: Unable to read file @ {geo_file}')

    if key_name in geodf.columns:
        groups = list(geodf[key_name].unique())
    else:
        print(f'[red]Error[\red]: {key_name} not found in geometry groups')

    sample_groups = []

    for group in groups:
        geometry_group = geodf[geodf[key_name] == group]

        grid = partition_geometry_group(
            geometry_group=geometry_group,
            n_sample_ims=n_sample_ims,
            spatial_size=spatial_size,
            seed=seed
        )
        grid = grid.set_crs(geodf.crs)
        grid['group'] = group
        grid['sample_id'] = grid.index

        sample_groups.append(grid)

    final_sample = pd.concat(sample_groups,ignore_index=True)
    final_sample = gpd.GeoDataFrame(final_sample,geometry=final_sample.geometry,crs=sample_groups[0].crs)

    if not splits:
        out_filename = Path(out_file) / f'{dataset_name}_{key_name}_sampling_{spatial_size_pixels}px_{n_sample_ims}im.geojson'
        final_sample.to_file(out_filename,driver='GeoJSON')
    else:
        assert (sum(split_sizes) == 1) and (len(splits) == len(split_sizes))   
        for split_prefix, split_size in zip(splits,split_sizes):
            split = final_sample.groupby('group').sample(n=min(int(split_size*n_sample_ims),len(final_sample)),random_state=seed)
            final_sample = final_sample.drop(split.index)   

            out_filename = Path(out_file) / f'{dataset_name}_{split_prefix}_{key_name}_sampling_{spatial_size_pixels}px_{n_sample_ims}im.geojson'
            split.to_file(out_filename,driver='GeoJSON')
    return out_filename


def partition_geometry_group(geometry_group,spatial_size,n_sample_ims,seed):
    ugeo = union(geometry_group.geometry)

    grid = gpd.GeoSeries(partition(ugeo.iloc[0],spatial_size))

    grid = grid.sample(n=n_sample_ims,random_state=seed)

    grid = gpd.GeoDataFrame(geometry=grid)

    return grid

    