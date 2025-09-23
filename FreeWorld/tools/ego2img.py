from nuscenes.nuscenes import NuScenes
from FreeWorld.map_expansion.map_api import FreeWorldMap # export PYTHONPATH=$PYTHONPATH:$(pwd)

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

data_root = '/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/FreeWorld'
map_name = "AIR_F11"
render_out_path = "ego2img" + "_" + map_name + ".png"
nusc = NuScenes(dataroot=data_root, version='v1.0-trainval', verbose=False)
nusc_map_bos = FreeWorldMap(dataroot=data_root, map_name=map_name, map_dim=FreeWorld_maps[map_name])

scene_tokens = [
    scene['token'] for scene in nusc.scene
    if nusc.get('log', scene['log_token'])['location'] == map_name
]

input_scene_tokens = scene_tokens
# input_scene_tokens = scene_tokens[0]  # only vis one scene

if not input_scene_tokens:
    print(f"No scenes found for location: {map_name}")
else:
    print(f"Found {len(scene_tokens)} scenes for location: {map_name}")

# 渲染 ego 位置并保存图像
ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(
    nusc,
    scene_tokens=input_scene_tokens,
    verbose=False,
    out_path=render_out_path,
    render_egoposes=False,
    render_egoposes_range=False,
    render_legend=False
)
