# Init nuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
from FreeWorld.map_expansion.map_api import FreeWorldMap  # export PYTHONPATH=$PYTHONPATH:$(pwd)

import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

FreeWorld_maps = {
    #'office_issac': (30, 87.0),  # 名称 'office_issac' 对应的地图尺寸 (width, height)
    
    'full_warehouse': (36.910000000000004, 74.99),
    'AIR_F1': (93.45, 47.300000000000004),
    'AIR_F11': (52.800000000000004, 58.400000000000006),
    'AIR_B1': (33.35, 48.2),
    'AIR_G': (34.6, 32.15)
}

data_root = 'data/FreeWorld'
map_name = "AIR_B1"
render_out_path = "map2img" + "_" + map_name + ".png"
nusc = NuScenes(dataroot=data_root, version='v1.0-trainval', verbose=False)

nusc_map = FreeWorldMap(dataroot=data_root, map_name=map_name, map_dim=FreeWorld_maps[map_name])

# render the map on front camera image.
sample_token = nusc.sample[9]['token']
layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']  # 只能渲染polygon
camera_channel = 'CAM_FRONT'
nusc_map.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel, out_path=render_out_path)


# Render ego poses.
# nusc_map_bos = NuScenesMap(dataroot='/Users/lau/data_sets/nuscenes', map_name='boston-seaport')
# ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[nusc.scene[1]['token']], verbose=False)