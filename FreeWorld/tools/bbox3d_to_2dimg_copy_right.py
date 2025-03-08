import numpy as np
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import os

# 加载NuScenes数据集
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/20250123_115447', verbose=True)

# 获取第一个sample
my_sample = nusc.sample[31]  # 获取样本
ann_list = my_sample['anns']  # 获取样本注释

# 获取相机前视图的sample_data
sample_data_token = my_sample['data']['CAM_BACK']
sd_rec = nusc.get('sample_data', sample_data_token)

# 获取图像文件路径并加载图像
filename = sd_rec['filename']
img_path = os.path.join(nusc.dataroot, filename)
img = cv2.imread(img_path)

# 确保当前数据为相机数据，并且是关键帧
assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
if not sd_rec['is_key_frame']:
    raise ValueError('The 2D re-projections are available only for keyframes.')

# 获取相关的sample和坐标系记录
s_rec = nusc.get('sample', sd_rec['sample_token'])
cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

# 获取样本中所有的annotation
ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]

# 处理每个annotation的3D框
for ann in ann_recs:
    # 获取目标的3D边界框信息
    box = ann['translation']  # 世界坐标系中的中心位置
    quat = Quaternion(ann['rotation'])  # 旋转四元数
    size = ann['size']  # 获取目标的大小 (length, width, height)

    # 打印初始的世界坐标系下的3D边界框
    print("==== World Coordinate ====")
    print(f"Ego translation:{pose_rec['translation']}")
    print(f"Ego rotation:{pose_rec['rotation']}")
    print(f"Box center (World): {box}")
    print(f"Box orientation (World): {quat}")
    print(f"Box size (World): {size}")

    # 1. 将目标从世界坐标系转换到自车坐标系
    ego_translation = np.array(pose_rec['translation'])
    ego_rotation = Quaternion(pose_rec['rotation'])

    # 计算从世界坐标系到自车坐标系的变换矩阵
    world_to_ego_translation = ego_translation
    world_to_ego_rotation = ego_rotation.rotation_matrix  # 旋转矩阵

    # 将3D边界框从世界坐标转换到自车坐标系
    box_center_world = np.array(box)  # 世界坐标中的3D边界框中心
    # 先进行平移，再进行旋转
    box_center_ego = world_to_ego_translation - box_center_world  # 先进行平移
    box_center_ego = np.dot(world_to_ego_rotation, box_center_ego)  # 再进行旋转

    # 打印自车坐标系下的3D边界框
    print("==== Ego (Vehicle) Coordinate ====")
    box_center_ego = [-3.9, 0.32, 0.29]
    #quat = [0, 0, 0, 1]
    print(f"Box center (Ego): {box_center_ego}")
    print(f"Box orientation (Ego): {quat}")
    print(f"Box size (Ego): {size}")

    # 2. 将自车坐标系中的3D边界框转换到相机坐标系
    camera_translation = np.array(cs_rec['translation'])
    camera_rotation = Quaternion(cs_rec['rotation']).rotation_matrix
    
    # 计算从自车坐标系到相机坐标系的变换
    ego_to_camera_translation = camera_translation
    ego_to_camera_rotation = camera_rotation
    
    # 将自车坐标系的3D边界框转换到相机坐标系
    box_center_camera = np.dot(ego_to_camera_rotation, box_center_ego - ego_to_camera_translation)

    # 打印相机坐标系下的3D边界框
    print("==== Camera Coordinate ====")
    print(f"Box center (Camera): {box_center_camera}")
    print(f"Box orientation (Camera): {quat}")
    print(f"Box size (Camera): {size}")

    # 3. 使用相机内参将3D坐标投影到2D图像坐标系
    # 假设每个3D框是一个立方体（8个角点）
    length, width, height = size  # 获取长宽高数据
    # 构建相对中心的8个角点（box size）
    corners_3d = np.array([
        [-length / 2, -width / 2, height / 2], [ length / 2, -width / 2, height / 2],
        [ length / 2,  width / 2, height / 2], [-length / 2,  width / 2, height / 2],
        [-length / 2, -width / 2, -height / 2], [ length / 2, -width / 2, -height / 2],
        [ length / 2,  width / 2, -height / 2], [-length / 2,  width / 2, -height / 2]
    ])  # 8个角点的相对位置

    # 旋转矩阵应用于每个角点，并加上相机中心的位置
    corners_3d_camera = np.dot(ego_to_camera_rotation, corners_3d.T).T + box_center_camera

    # 投影到2D图像坐标
    box_img = []
    for corner in corners_3d_camera:
        x, y, z = corner  # 取每个角点
        if z > 0:  # 确保Z大于0以避免除零错误
            u = camera_intrinsic[0, 0] * (x / z) + camera_intrinsic[0, 2]
            v = camera_intrinsic[1, 1] * (y / z) + camera_intrinsic[1, 2]
            box_img.append([u, v])

    # If no valid 2D points were added to box_img, we cannot proceed with reshaping
    if len(box_img) == 0:
        print("Warning: No valid 2D projections found (all points behind camera). Skipping this annotation.")
        continue

    box_img = np.array(box_img).T  # 2x8 的数组，存储每个角点的2D投影坐标

    # Ensure box_img is 2D and has shape (2, 8)
    if len(box_img.shape) == 1:
        box_img = np.reshape(box_img, (2, 8))

    # 遍历绘制3D框的各个边
    color = (64, 128, 255)  # 绿色边框
    thickness = 1
    for i in range(4):
        j = (i + 1) % 4
        # 绘制下底面的边
        cv2.line(img, (int(box_img[0, i]), int(box_img[1, i])), (int(box_img[0, j]), int(box_img[1, j])), color, thickness=thickness)
        # 绘制上底面的边
        cv2.line(img, (int(box_img[0, i + 4]), int(box_img[1, i + 4])), (int(box_img[0, j + 4]), int(box_img[1, j + 4])), color, thickness=thickness)
        # 绘制侧边的边
        cv2.line(img, (int(box_img[0, i]), int(box_img[1, i])), (int(box_img[0, i + 4]), int(box_img[1, i + 4])), color, thickness=thickness)

# 保存结果图像
cv2.imwrite('output_image.jpg', img)
print("记录好了图片")
print(f"六个视角总的annotation数量：{len(ann_list)}")
