from pathlib import Path

import geopandas as gpd
import rasterio as rio

from hydra.conf import HydraConf
from omegaconf import OmegaConf, DictConfig
from hydra import initialize,compose
from hydra.utils import instantiate, call

def load_model(model_path): 
    pass

def run_inference(self,args):
    pass