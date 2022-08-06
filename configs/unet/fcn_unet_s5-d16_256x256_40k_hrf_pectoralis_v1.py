_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/my_pectoralis.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_cumulative.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
    auxiliary_head=dict(norm_cfg=norm_cfg))

img_norm_cfg = dict(
    mean=[21.601, 21.601, 21.601], std=[43.724, 43.724, 43.724], to_rgb=True)


img_scale = (256, 256)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(type='Resize', img_scale=(1024, 1024)),
    # dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),

    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='CLAHE', clip_limit=3),
    dict(type='AdjustGamma'),
    dict(type='RandomRotate', prob=0.5, degree=(90,90)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
            # dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


model = dict(
    # pretrained='',
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    # test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
    test_cfg=dict(mode='whole'))
evaluation = dict(metric='mDice')
