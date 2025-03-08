import os.path

import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval',dataroot = 'data/FreeWorld',verbose=True)

my_sample = nusc.sample[93]  # 得到一个sample
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

    # 世界坐标系下的box标注信息
    box = nusc.get_box(ann_rec['token'])
    print("------------------------------------------------------------")
    print("本车世界坐标系 translation:", np.array(pose_rec['translation']))
    print("本车世界坐标系 rotation:", Quaternion(pose_rec['rotation']))
    print("box世界坐标系 translation:", box.center)
    print("box世界坐标系 rotation:", box.orientation)
    print("------------------------------------------------------------")
    # box.center = [2.1, 0.5, 0.2]
    # box.orientation = Quaternion([0, 0, 0, 1])
    # box.whl = [0.48, 0.76, 1.80]

    # 从世界坐标系->车身坐标系
    box.translate(-np.array(pose_rec['translation']))
    box.rotate(Quaternion(pose_rec['rotation']).inverse)
    print("车辆坐标系下box translation:", box.center)
    print("车辆坐标系下box rotation:", box.orientation)
    print("------------------------------------------------------------")
    
    # box.center = [-3.9, 0.32, 0.7]
    # box.orientation = Quaternion([0, 0, 0, 1])
    # box.whl = [0.48, 0.76, 2.10]

    # 从车身坐标系->相机坐标系
    # cam2lidar back
    #cs_rec['translation'] = [0, 0.112, -0.105]
    #cs_rec['rotation'] = [0.5, -0.5, -0.5, 0.5]
    
    box.translate(-np.array(cs_rec['translation']))
    box.rotate(Quaternion(cs_rec['rotation']).inverse)
    print("相机坐标系下box translation:", box.center)
    print("相机坐标系下box rotation:", box.orientation)
    print("------------------------------------------------------------")

    # 过滤掉不在不在图像传感器前面的点
    corners_3d = box.corners()
    # print('+++++++++++++++++++++++++++++', corners_3d)
    # in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    # corners_3d = corners_3d[:, in_front]
    #print('------------------------------', corners_3d)
    # 从相机坐标系->像素坐标系
    view = np.eye(4)
    view[:3, :3] = np.array(camera_intrinsic)
    # in_front = corners_3d[2, :] > 0.1
    # if all(in_front) is False:
    #     continue
    points = corners_3d
    points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
    points = np.dot(view, points)[:3, :]
    points /= points[2, :]

    box_img = points.astype(np.int32)
    color = (64, 128, 255)
    # print(box_img)
    b = list(set(ann_rec['category_name'].split('.')).intersection(track_class))
    if len(b) == 0:
        continue
    else:
        for i in range(4):
            j = (i + 1) % 4
            # 下底面
            cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, j], box_img[1, j]), color, thickness=1)
            # 上底面
            cv2.line(img, (box_img[0, i + 4], box_img[1, i + 4]), (box_img[0, j + 4], box_img[1, j + 4]), color, thickness=1)
            # 侧边线
            cv2.line(img, (box_img[0, i], box_img[1, i]), (box_img[0, i + 4], box_img[1, i + 4]), color, thickness=1)

# 保存结果图像
cv2.imwrite('output_image.jpg', img)
print("记录好了图片")
print(f"六个视角总的annotation数量：{len(ann_list)}")
