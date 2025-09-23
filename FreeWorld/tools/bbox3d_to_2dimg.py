import os.path

import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval',dataroot = 'data/NuscenesData',verbose=True)

my_sample = nusc.sample[2]  # 得到一个sample
ann_list = my_sample['anns']  # 取出该sample的所有sample_annotation_token

sample_data_token = my_sample['data']['CAM_FRONT']

sd_rec = nusc.get('sample_data', sample_data_token)

filename = sd_rec['filename']
img_path = os.path.join(nusc.dataroot, filename)
img = cv2.imread(img_path)

assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
if not sd_rec['is_key_frame']:
    raise ValueError('The 2D re-projections are available only for keyframes.')

s_rec = nusc.get('sample', sd_rec['sample_token'])

cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]

track_class = {'car', 'truck', 'trailer', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}

for ann_rec in ann_recs:
    ann_rec['sample_annotation_token'] = ann_rec['token']
    ann_rec['sample_data_token'] = sample_data_token
    print("sample data token:", sample_data_token)

    # 世界坐标系下的box标注信息
    box = nusc.get_box(ann_rec['token'])
    
    # box.center = [11.9454331 ,  7.91601075,  0.22773173]
    # box.orientation = Quaternion(-0.027015588265131577, -0.0029624739646874664, -0.10896425997942423, 0.9936740772435567)
    # box.wlh = np.array([0.48, 0.76, 1.80])
    print("------------------------------------------------------------")
    print("本车世界坐标系 translation:", np.array(pose_rec['translation']))
    print("本车世界坐标系 rotation:", Quaternion(pose_rec['rotation']))
    print("box世界坐标系 translation:", box.center)
    print("box世界坐标系 rotation:", box.orientation)
    print("box的大小 size:", box.wlh)
    print("------------------------------------------------------------")

    # 从世界坐标系->车身坐标系
    box.translate(-np.array(pose_rec['translation']))
    box.rotate(Quaternion(pose_rec['rotation']).inverse)
    #box.center = [8.74, -5.83, 0.3405]
    #box.center[2] = -0.8
    # box.orientation = [0, 0, 0, 1]
    print("车辆坐标系下box translation:", box.center)
    print("车辆坐标系下box rotation:", box.orientation)
    print("------------------------------------------------------------")
    
    # box.center = [-2.028576874592424, 1.74543679954891, 0.16528985829391607]
    # box.orientation = [0.2377,−0.0070,0.0067,−0.9713]

    
    # box.center = [-3.9, 0.32, 0.7]
    # box.orientation = Quaternion([0, 0, 0, 1])
    # box.whl = [0.48, 0.76, 2.10]
        
    # 从车身坐标系->相机坐标系
    # cam2lidar back
    # cs_rec['translation'] = [-0.090916, 0.112, 0.0524717]
    # cs_rec['rotation'] = [0.683, -0.683, -0.183, 0.183]
    box.translate(-np.array(cs_rec['translation']))
    box.rotate(Quaternion(cs_rec['rotation']).inverse)
    print("相机坐标系下box translation:", box.center)
    print("相机坐标系下box rotation:", box.orientation)
    print("------------------------------------------------------------")

    # 获取所有8个角点
    corners_3d = box.corners()
    
    # 检查是否有足够的点在图像前方 (z > 0)
    visible_points = np.sum(corners_3d[2, :] > 0)
    
    # 如果可见点不足4个，则跳过此box的绘制
    if visible_points < 4:
        print(f"跳过: 只有 {visible_points}/8 个点可见")
        continue
    
    # 转换所有8个角点到图像坐标，即使有些在相机后面
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    
    # 复制一份所有角点
    all_corners_3d = corners_3d.copy()
    
    # 把在相机后面的点投影到相机前方的某个位置（这里用一个很小的正值）
    all_corners_3d[2, all_corners_3d[2, :] <= 0] = 0.1
    
    points = all_corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]
    
    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    
    b = list(set(ann_rec['category_name'].split('.')).intersection(track_class))
    if len(b) == 0:
        continue
    
    # 绘制3D框，但只有当应该看到的边时才绘制
    edges = [
        # 底部面
        (0, 1), (1, 2), (2, 3), (3, 0),
        # 顶部面
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 连接线
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for start, end in edges:
        # 只绘制当两个端点都在相机前面时的边
        if corners_3d[2, start] > 0 and corners_3d[2, end] > 0:
            cv2.line(img, 
                    (box_img[0, start], box_img[1, start]), 
                    (box_img[0, end], box_img[1, end]), 
                    color, thickness=1)
    # break

# 保存结果图像
cv2.imwrite('output_image.jpg', img)
print("记录好了图片")
print(f"六个视角总的annotation数量：{len(ann_list)}")
