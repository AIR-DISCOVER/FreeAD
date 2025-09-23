
# train with 8 card
#**NOTE**: We release two types of training configs: the end-to-end configs and the two-stage (stage-1: Perception & Prediction; stage-2: Planning) configs. They should produce similar results. 
#The two-stage configs are recommended because you can just train the stage-1 model once and use it as a pre-train model for stage-2.
#最好选择两个阶段的config，一个阶段一个阶段的训练，前面的可以作为pretrained model
export CUDA_VISIBLE_DEVICES="0,1"  # 在 多卡的情况下指定
#python -m torch.distributed.run --nproc_per_node=3 --master_port=2734 tools/train.py projects/configs/FreeAD/FreeAD_base_map_fine_tune.py --launcher pytorch --deterministic --work-dir path/to/save/outputs

python -m torch.distributed.run --nproc_per_node=2 --master_port=2734 tools/train.py projects/configs/FreeAD/FreeAD_base_stage_2.py --launcher pytorch --deterministic --work-dir path/to/save/outputs/FreeeAskWorlds1e3s2e1_FreeWorld_s1e1s2e1

# train with 1 card
#python tools/train.py projects/configs/VAD/FreeAD_base.py --deterministic --work-dir path/to/save/outputs


# test
#python tools1/check_train_data.py projects/configs/VAD/VAD_base_issac.py --deterministic --work-dir path/to/save/outputs