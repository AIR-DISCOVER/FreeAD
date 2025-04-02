import json

input_filename = "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/FreeWorld/v1.0-trainval/sample_data.json"   # 输入 JSON 文件名
output_filename = "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/data/FreeWorld/v1.0-trainval/sample_data_fix.json" # 输出 JSON 文件名

# 从文件中读取 JSON 数据
with open(input_filename, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# 遍历 JSON 数据，替换每个对象中 filename 的所有 ":" 为 "_"
for item in data:
    if "filename" in item:
        item["filename"] = item["filename"].replace(":", "_")

# 将修改后的 JSON 数据写回到文件中
with open(output_filename, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)
