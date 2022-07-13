_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/my_pectoralis.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
data_root = 'data/pectoralis_dataset_cropbg_unified'

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))

img_norm_cfg = dict(
    mean=[28.032, 28.032, 28.032], std=[47.868, 47.868, 47.868], to_rgb=True)


img_scale = (512, 320)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    # keep_ratio 参数，默认是 True
    # 当 keep_ratio 设置为 False 时候，输入的 img_scale 含义是 w,h, 输出尺度就是预设的输入尺度。
    # 当 keep_ratio 设置为 True 时候，输入的 img_scale 不区分 h,w，
    # 而是 min 和 max，输出会在保证宽高比不变的情况下，最大程度接近 [min(img_scale), max(img_scale)] 范围。
    # dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(type='Resize', img_scale=(1024, 1024)),

    # 多尺度，ratio_range 模式，随机采样 ratio_range 范围，然后作用于 img_scale 上
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='Resize', img_scale=img_scale),

    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='CLAHE', clip_limit=3),
    dict(type='AdjustGamma'),
    # dict(type='RandomRotate', prob=0.5, degree=(90,90)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks',
        pipeline=train_pipeline),
    val=dict(
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks',
        pipeline=test_pipeline),
    test=dict(
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks',
        pipeline=test_pipeline))


model = dict(
    # pretrained='',
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
evaluation = dict(metric='mIoU')
