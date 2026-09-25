from pathlib import Path

import numpy as np
import geopandas as gpd

import rasterio as rio

import torch
from torch.utils.data import Dataset


class LandfireDataset(Dataset):
    FM40_LABELS = [
        91, 92, 93, 98, 99,
        101, 102, 103, 104, 105, 106, 107, 108, 109,
        121, 122, 123, 124,
        141, 142, 143, 144, 145, 146, 147, 148, 149,
        161, 162, 163, 164, 165,
        181, 182, 183, 184, 185, 186, 187, 188, 189,
        201, 202, 203, 204,
    ]

    def __init__(
            self,
            data_root,
            geometry_root,
            img_size=128,
            split='train',
            dataset_name='fuelex',
            group_name='superzone',
            groups=['Northwest'],
            ims_per_group=3000,
            year=2025,
            label_dataset='fbfm40',
            transform=None,
            ignore_index=-1,
            dropped_labels=[91,92,93,98,99]
        ):
        super().__init__()

        self.data_root = Path(data_root)
        self.geometry_root = Path(geometry_root)
        self.img_size = img_size
        self.split = split
        self.dataset_name=dataset_name
        self.group_name = group_name
        self.groups = groups
        self.ims_per_group = ims_per_group
        self.year = year
        self.label_dataset = label_dataset
        self.transform = transform
        self.ignore_index = ignore_index
        self.dropped_labels = dropped_labels

        self.geometry_path = Path(geometry_root) / f'{dataset_name}_{split}_{group_name}_sampling_{img_size}px_{ims_per_group}im.geojson'
        self.geometries = gpd.read_file(self.geometry_path)
        self.geometries = self.geometries[self.geometries['group'].isin(self.groups)]

        self.valid_labels = [label for label in self.FM40_LABELS if not any(label == ignored for ignored in self.dropped_labels)]
        self.classes = self.valid_labels
        self.n_classes = len(self.valid_labels)
        self.label_encode = self.make_label_encoder()

    def make_label_encoder(self):
        self.label_map = dict(zip(self.valid_labels,np.arange(len(self.valid_labels))))
        self.label_map[-1] = self.ignore_index
            

        def encode_fn(x):
            return self.label_map[x]
        
        label_encode = np.vectorize(encode_fn)
        
        return label_encode

    def labels(self):
        return self.valid_labels

    def __getitem__(self, index):
        sample = self.geometries.iloc[index]

        aef_fname = self.data_root / f'AEF_{self.year}_Scene{sample['sample_id']}.tif'
        aef_arr = rio.open(aef_fname).read()


        label_fname = self.data_root / f'{self.label_dataset.upper()}_{self.year}_Scene{sample['sample_id']}.tif'
        label_arr = rio.open(label_fname).read()

        label_arr[np.isin(label_arr,np.array(self.dropped_labels))] = self.ignore_index
        label_arr = self.label_encode(label_arr)
        
        aef_tensor = torch.from_numpy(aef_arr.astype(np.float64)).float()
        label_tensor = torch.from_numpy(label_arr.astype(np.int64)).long()

        if self.transform:
            aef_tensor = self.transform(aef_tensor)

        output = {
            'input':{
                'aef': aef_tensor
            },
            'target':label_tensor
        }
        return output

    def __len__(self):
        return len(self.geometries)