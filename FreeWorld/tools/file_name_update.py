import json

sample_data_json_path = "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/FreeWorld/v1.0-trainval/sample_data.json"

# 读取 JSON 文件
with open(sample_data_json_path, 'r') as file:
    data = json.load(file)

# 只修改 'filename' 键中的值
for item in data:
    if 'filename' in item:
        item['filename'] = item['filename'].replace(":", "_")

# 将修改后的数据保存回文件
with open(sample_data_json_path, 'w') as file:
    json.dump(data, file, indent=4)

print("文件已更新，'filename' 中的冒号已替换为下划线")
