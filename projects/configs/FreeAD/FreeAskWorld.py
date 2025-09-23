# 加载自定义的数据集配置文件和默认的运行时配置文件
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# 启用插件功能，并指定插件目录路径
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 设置点云的范围和体素大小，适用于3D检测任务
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 图像归一化配置，用于数据预处理
# img_norm_cfg = dict(
#    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
   mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)


# maps roscenes have, map_dims 在map的canvas_edge里面[0]宽[1]长，数值单位米
FreeWorld_maps = {
    #'office_issac': (30, 87.0),  # 名称 'office_issac' 对应的地图尺寸 (width, height)
    
    'full_warehouse': (36.910000000000004, 74.99),
    'AIR_F1': (93.45, 47.300000000000004),
    'AIR_F11': (52.800000000000004, 58.400000000000006),
    'AIR_B1': (33.35, 48.2),
    'AIR_G': (34.6, 32.15),
    'HongKongCity': (100, 180)
}

# For nuScenes we usually do 10-class detection
# 定义检测的类别名称和类别数量
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
# 定义地图的类别及相关参数，用于地图生成任务
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0  # 定义真实和预测线段上固定的点数
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = False  # 设置评估时是否使用相同数量的真实标注样本
map_num_classes = len(map_classes)  

# 输入模式配置，设置模型只使用相机数据，而不使用激光雷达、雷达和地图数据
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    #use_external=True)
    use_external=False)

# 模型参数设置，包括维度、BEV（鸟瞰图）大小、序列长度和总训练轮数
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
total_epochs = 12

model = dict(
    # 定义模型的主干（ResNet50）用于特征提取
    # 中间层金字塔网络（FPN）用于特征调整
    type='VAD',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),

    # 定义点云检测头，包括各种门限和特定配置    
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=total_epochs,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        # 自定义解码器，用于处理自车（ego）的历史轨迹
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        # 自定义解码器，用于处理移动轨迹
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        # 配置边界框编码器，用于3D检测的预测框生成
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        # 损失函数配置，包括规划、碰撞检测等的损失权重
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0),
        loss_bbox=dict(type='L1Loss', loss_weight=0),
        loss_traj=dict(type='L1Loss', loss_weight=0),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=10),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=0),
            pc_range=point_cloud_range))))

# 数据集类型和数据根目录配置
dataset_type = 'FreeWorldDataset'
data_root = 'data/FreeAskWorld/'
file_client_args = dict(backend='disk')

# 训练数据处理流程，包括加载图像、多视图畸变、数据归一化和增强、打包格式等
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

# 测试数据处理流程，包含与训练类似的处理步骤和数据增强
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels'])])
]

# 定义训练数据集，包括标注文件、类别和预处理管道
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # this pass to the dataset builder
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vad_freeaskworld_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        maps=FreeWorld_maps,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    # 定义验证数据集
    val=dict(type=dataset_type,
             data_root=data_root,
             pc_range=point_cloud_range,
             ann_file=data_root + 'vad_freeaskworld_infos_temporal_train.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             maps=FreeWorld_maps,
             map_classes=map_classes,
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    # 定义测试数据集
    test=dict(type=dataset_type,
              data_root=data_root,
              pc_range=point_cloud_range,
              ann_file=data_root + 'vad_freeaskworld_infos_temporal_train.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              maps=FreeWorld_maps,
              map_classes=map_classes,
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# 优化器配置，使用 AdamW 优化器和特定学习率
# paramwise_cfg润许为特定的层设置不同学习率，也可以冻结模型
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 微调修改学习率，之前是-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0),
            'img_neck':dict(lr_mult=0),
            'img_backbone': dict(lr_mult=0),
            'pts_bbox_head.ego_agent_decoder':dict(lr_mult=0),
            'pts_bbox_head.ego_map_decoder':dict(lr_mult=0),
            'pts_bbox_head.motion_decoder':dict(lr_mult=0),
            'pts_bbox_head.bbox_coder':dict(lr_mult=0),
            'pts_bbox_head.map_bbox_coder':dict(lr_mult=0),
            'pts_bbox_head.positional_encoding':dict(lr_mult=0),
        }),
    weight_decay=0.01)
# optimizer = dict(
#     type='AdamW',
#     lr=2e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)

# 学习率调度器配置，使用余弦退火策略并包含线性预热
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# 评估设置，使用bbox和Chamfer距离作为评估指标
evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

# 模型训练设置，基于epoch的runner并加载预训练模型
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
#load_from = 'path/to/save/expert/epoch_col36.pth'
load_from = "/home/tsinghuaair/pengyh/FreeAD/my_project/FreeAD/ckpts/VAD_base.pth"

# 日志配置，每100步记录一次，支持文本和Tensorboard日志
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.)
# find_unused_parameters = True

# 检查点配置，设置检查点保存间隔和保留的最大检查点数
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

# 自定义钩子，用于设置和记录epoch信息
custom_hooks = [dict(type='CustomSetEpochInfoHook')]