
# 设置变量
DataSet="FreeWorld"
ROOT_PATH="./data/$DataSet"
OUT_DIR="./data/$DataSet"
EXTRA_TAG="vad_nuscenes"
VERSION="v1.0"
CANBUS_PATH="./data/$DataSet/can_bus"

export PYTHONPATH=$PYTHONPATH:/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD
# 提取数据集，生成pkl文件
python tools/data_converter/freead_data_converter.py nuscenes --root-path "$ROOT_PATH" --out-dir "$OUT_DIR" --extra-tag "$EXTRA_TAG" --version "$VERSION" --canbus "$CANBUS_PATH" 
