import os
import shutil
import json
from collections import defaultdict

'''
    This script merges data from separate directories while maintaining the structure.
'''

# 加载JSON数据的函数
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 合并map数据的函数，根据filename合并log_tokens
def merge_map_data(folders):
    merged_map = defaultdict(lambda: {
        "category": None,
        "filename": None,
        "log_tokens": [],
        "token": None
    })

    # 遍历每个文件夹，合并map数据
    for folder in folders:
        map_path = os.path.join(folder)  # Ensure correct path to maps folder
        if os.path.exists(map_path):
            for map_file in os.listdir(map_path):
                if map_file.endswith('map.json'):
                    map_data = load_json(os.path.join(map_path, map_file))
                    for map_entry in map_data:
                        # Check if 'filename' exists in map_entry before proceeding
                        if 'filename' not in map_entry:
                            print(f"Skipping map entry due to missing 'filename': {map_entry}")
                            continue
                        
                        filename = map_entry["filename"]
                        
                        if filename in merged_map:
                            merged_map[filename]["log_tokens"].extend(map_entry["log_tokens"])
                        else:
                            merged_map[filename]["category"] = map_entry.get("category", None)
                            merged_map[filename]["filename"] = map_entry.get("filename", None)
                            merged_map[filename]["token"] = map_entry.get("token", None)
                            merged_map[filename]["log_tokens"] = map_entry.get("log_tokens", [])

    # 将defaultdict转换为列表返回
    merged_map_list = list(merged_map.values())
    return merged_map_list

# 合并can_bus数据的函数，根据timestamp合并数据
def merge_can_bus_data(folders):
    merged_can_bus = defaultdict(lambda: {"data": []})

    # 遍历每个文件夹，合并can_bus数据
    for folder in folders:
        can_bus_path = os.path.join(folder, 'can_bus', 'can_bus.json')
        if os.path.exists(can_bus_path):
            can_bus_data = load_json(can_bus_path)
            for timestamp, data in can_bus_data.items():
                merged_can_bus[timestamp]["data"].extend(data["data"])

    # 转换为列表返回
    return dict(merged_can_bus)

# 合并sweeps和samples文件夹的函数，直接复制整个文件夹
def merge_sweeps_samples(folders, output_dir):
    sweeps_output_dir = os.path.join(output_dir, 'sweeps')
    samples_output_dir = os.path.join(output_dir, 'samples')
    
    # 确保输出目录存在
    os.makedirs(sweeps_output_dir, exist_ok=True)
    os.makedirs(samples_output_dir, exist_ok=True)
    
    # 遍历每个文件夹，合并sweeps和samples文件夹
    for folder in folders:
        sweeps_path = os.path.join(folder, 'sweeps')
        samples_path = os.path.join(folder, 'samples')
        
        # 复制sweeps文件夹的内容
        if os.path.exists(sweeps_path):
            for item in os.listdir(sweeps_path):
                item_path = os.path.join(sweeps_path, item)
                if os.path.isdir(item_path):
                    dest_path = os.path.join(sweeps_output_dir, item)
                    shutil.copytree(item_path, dest_path, dirs_exist_ok=True)
        
        # 复制samples文件夹的内容
        if os.path.exists(samples_path):
            for item in os.listdir(samples_path):
                item_path = os.path.join(samples_path, item)
                if os.path.isdir(item_path):
                    dest_path = os.path.join(samples_output_dir, item)
                    shutil.copytree(item_path, dest_path, dirs_exist_ok=True)

# 合并所有JSON文件的函数
def merge_json_files(folders, output_dir):
    merged_data = {
        "can_bus": {},
        "ego_pose": [],
        "instance": [],
        "log": [],
        "sample": [],
        "sample_data": [],
        "sample_annotation": [],
        "scene": [],
        "map": [],
    }

    # 遍历每个文件夹
    for folder in folders:
        print(f"正在处理文件夹: {folder}")
        
        # 定义需要合并的文件路径
        can_bus_path = os.path.join(folder, 'can_bus', 'can_bus.json')
        ego_pose_path = os.path.join(folder, 'ego_pose.json')
        instance_path = os.path.join(folder, 'instance.json')
        log_path = os.path.join(folder, 'log.json')
        sample_path = os.path.join(folder, 'sample.json')
        sample_data_path = os.path.join(folder, 'sample_data.json')
        sample_annotation_path = os.path.join(folder, 'sample_annotation.json')
        scene_path = os.path.join(folder, 'scene.json')

        # 检查文件是否存在，如果存在则加载并合并
        if os.path.exists(can_bus_path):
            merged_data["can_bus"].update(merge_can_bus_data([folder]))
        if os.path.exists(ego_pose_path):
            merged_data["ego_pose"].extend(load_json(ego_pose_path))
        if os.path.exists(instance_path):
            merged_data["instance"].extend(load_json(instance_path))
        if os.path.exists(log_path):
            merged_data["log"].extend(load_json(log_path))
        if os.path.exists(sample_path):
            merged_data["sample"].extend(load_json(sample_path))
        if os.path.exists(sample_data_path):
            merged_data["sample_data"].extend(load_json(sample_data_path))
        if os.path.exists(sample_annotation_path):
            merged_data["sample_annotation"].extend(load_json(sample_annotation_path))
        if os.path.exists(scene_path):
            merged_data["scene"].extend(load_json(scene_path))

    # 单独合并map数据
    merged_data["map"] = merge_map_data(folders)

    # 合并sweeps和samples文件夹
    merge_sweeps_samples(folders, output_dir)
    
    return merged_data

# 将合并后的数据保存到指定目录的JSON文件
def save_merged_data(merged_data, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存can_bus数据到can_bus文件夹
    can_bus_output_dir = os.path.join(output_dir, 'can_bus')
    os.makedirs(can_bus_output_dir, exist_ok=True)
    with open(os.path.join(can_bus_output_dir, 'can_bus.json'), 'w') as f:
        json.dump(merged_data["can_bus"], f, indent=4)
    print(f"can_bus 数据已保存至: {can_bus_output_dir}/can_bus.json")

    # 保存其他数据（非can_bus）到合并目录
    for key, data in merged_data.items():
        if key != "can_bus":
            output_path = os.path.join(output_dir, f'{key}.json')
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"{key} 数据已保存至: {output_path}")

# 输入需要合并的文件夹路径（请根据实际情况替换）
data_root = 'data/AIR_B1_修复_标注完成/'
folders = [
    "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/AIR_B1_修复_标注完成/merged_data",
    "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/AIR_F1_修复_标注完成/AIR_F1_修复/merged_data",
    "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/AIR_F11_epoch_2_修复_标注完成/AIR_F11_epoch_2_修复/merged_data",
    "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/AIR_G_修复_标注完成/merged_data"
    #data_root + '20250122_192300',
    #data_root + '20250122_192413',
    #data_root + '20250122_192520',
    #data_root + '20250122_192629',
    #data_root + '20250123_154657',
    #data_root + '20250123_154818',
    #data_root + 'NuscenesData'
    # data_root + '20250123_120533',
    # data_root + '20250123_120637',
    # data_root + '20250123_120804',
    # data_root + '20250123_120911',
    # data_root + 'NuscenesData'
]

# 指定合并后数据保存的输出目录
output_dir = data_root + "merged_data"

# 合并数据并保存
merged_data = merge_json_files(folders, output_dir)
save_merged_data(merged_data, output_dir)
