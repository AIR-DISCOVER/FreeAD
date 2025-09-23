import json
import numpy as np
from pyquaternion import Quaternion

def rotate_quaternion_around_z(quat, angle_degrees):
    """
    将四元数绕Z轴旋转指定角度
    
    参数:
    quat: 原始四元数，格式为[w, x, y, z]
    angle_degrees: 旋转角度(度)，正值表示逆时针旋转，负值表示顺时针旋转
    
    返回:
    new_quat: 旋转后的四元数，格式为[w, x, y, z]
    """
    # 创建原始四元数
    original_quat = Quaternion(quat[0], quat[1], quat[2], quat[3])
    
    # 创建绕Z轴旋转的四元数
    angle_rad = np.radians(angle_degrees)
    z_rotation = Quaternion(axis=[0, 0, 1], angle=angle_rad)
    
    # 应用旋转
    new_quat = z_rotation * original_quat
    
    return [new_quat.w, new_quat.x, new_quat.y, new_quat.z]

def process_json_file(json_path, z_rotate_angle, is_xyzw_to_wxyz):
    """
    读取JSON文件，绕Z轴旋转每个对象中的rotation四元数，并保存结果
    
    参数:
    json_path: JSON文件路径
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 确保数据是列表类型
    if not isinstance(data, list):
        print("JSON数据不是一个列表")
        return
    
    # 旋转角度
    angle = z_rotate_angle 
    
    # 遍历每个对象并旋转rotation
    for item in data:
        if 'rotation' in item and isinstance(item['rotation'], list) and len(item['rotation']) == 4:
            if is_xyzw_to_wxyz == True:
                # xyzw -> wxyz
                item['rotation'] = [
                    item['rotation'][3],  # w
                    item['rotation'][0],  # x
                    item['rotation'][1],  # y
                    item['rotation'][2]   # z
                ]

            item['rotation'] = rotate_quaternion_around_z(item['rotation'], angle)
    
    # 保存修改后的JSON数据
    output_path = json_path.replace('.json', '_rotated.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"处理完成。已将结果保存至 {output_path}")

# 使用示例
if __name__ == "__main__":
    dataset_root = "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/NuscenesData" + "/v1.0-trainval"
    
    ego_pose_json_path = dataset_root + "/ego_pose.json"  # 替换为你的JSON文件路径
    sample_annotation_json_path = dataset_root + "/sample_annotation.json"
    
    transfer_angle = -90
    process_json_file(ego_pose_json_path, transfer_angle, False)  # 不需要xyzw-->wxyz
    process_json_file(sample_annotation_json_path, transfer_angle, False)