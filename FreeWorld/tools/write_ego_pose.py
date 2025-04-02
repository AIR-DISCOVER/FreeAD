import json

# 读取JSON文件
input_file = '/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/FreeWorld/v1.0-trainval/ego_pose.json'  # 你的输入文件名
output_file = 'output.json'  # 输出文件名，保存修改后的数据

# 打开并加载 JSON 数据
with open(input_file, 'r') as file:
    data = json.load(file)

# 遍历每个字典对象，修改 'translation' 键的值
for item in data:
    if 'translation' in item:
        item['translation'] = [0, 0, 0]

# 将修改后的数据保存到新的JSON文件
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"修改后的数据已保存到 {output_file}")
