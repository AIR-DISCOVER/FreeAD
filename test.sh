
# 推理使用eval
#CUDA_VISIBLE_DEVICES=1 python tools/test.py projects/configs/FreeAD/FreeAD_base.py path/to/save/outputs/VAD_base.pth --launcher none --eval bbox --tmpdir tmp
#CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/FreeAD/FreeAD_base.py path/to/save/outputs/epoch_48.pth --launcher none --eval bbox --tmpdir tmp
#CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/FreeAD/FreeAD_base.py ckpts/3dbox数据修复_FreeAD_s1e6_s2e3.pth --launcher none --eval bbox --tmpdir tmp
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/FreeAD/FreeAskWorld.py ckpts/VAD_FreeAskWorlds1e3s2e1.pth --launcher none --eval bbox --tmpdir tmp
#CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/FreeAD/FreeAD_base.py ckpts/VAD_base.pth --launcher none --eval bbox --tmpdir tmp  
#CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/FreeAD/VAD_tiny_e2e.py ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

# 推理我自己生成pkl文件,直接out不eval，表示直接获取bbox等信息,out代表输出原始（没有eval）推理数据的,format_only代表不eval输出vis的格式数据
##CUDA_VISIBLE_DEVICES=0 python tools/test.py  projects/configs/FreeAD/FreeAD_base.py path/to/save/outputs/epoch_36.pth --launcher none --tmpdir tmp --out path/to/results/my_data_results_no_eval.pkl --format-only


#CUDA_VISIBLE_DEVICES=0 python tools/test.py  projects/configs/FreeAD/FreeAD_base.py ckpts/VAD_base.pth --launcher none --tmpdir tmp --out path/to/results/my_data_results_no_eval.pkl --format-onlys