import json
import random

def read_names_from_json(file_path):
    # 打开并读取 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)  # 加载 JSON 数据

    # 提取所有的 name 字段
    names = [entry['name'] for entry in data]

    return names

def split_dataset(names, train_ratio=0.7, val_ratio=0.3, test_ratio=0):
    # 确保总比例是1
    assert train_ratio + val_ratio + test_ratio == 1, "Train, val and test ratios must sum to 1"

    # 打乱数据集
    random.shuffle(names)

    # 计算分割的索引
    total_count = len(names)
    train_end = int(train_ratio * total_count)
    val_end = train_end + int(val_ratio * total_count)

    # 划分数据集
    train_set = names[:train_end]
    val_set = names[train_end:val_end]
    test_set = names[val_end:]

    return train_set, val_set, test_set

def map_names_to_split(names, train_set, val_set, test_set):
    # 为每个名字创建对应的映射
    name_to_split = {}
    for name in names:
        if name in train_set:
            name_to_split[name] = 'train'
        elif name in val_set:
            name_to_split[name] = 'val'
        else:
            name_to_split[name] = 'test'

    return name_to_split

def save_mapping_to_file(mapping, file_path):
    # 将映射保存为 JSON 文件
    with open(file_path, 'w') as file:
        json.dump(mapping, file, indent=4)

data_root = 'data/FreeWorld/'
file_path = data_root + 'v1.0-trainval/' + 'scene.json'  # 读取所有场景
names = read_names_from_json(file_path)

# 划分数据集
train_set, val_set, test_set = split_dataset(names)

# 创建名称到数据集的映射
name_to_split = map_names_to_split(names, train_set, val_set, test_set)

# 保存映射到文件
mapping_file_path = data_root + 'FreeWorldDataSetSplitMapping.json'  # 保存映射的文件路径
save_mapping_to_file(name_to_split, mapping_file_path)

print(f"Name to split mapping has been saved to {mapping_file_path}")

# 打印映射前几个数据集
print("Name to Split Mapping (first 5):", {k: name_to_split[k] for k in list(name_to_split)[:5]})
