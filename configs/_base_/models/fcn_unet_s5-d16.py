# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        # 输入图像通道数。 默认3。
        in_channels=3,
        # 每个阶段的基本通道数。/第一阶段的输出通道。默认值：64
        base_channels=64,
        # 编码器的级数，通常为5
        num_stages=5,
        # len(strides)等于num_stages，取值[int 1 | 2]，如果strides[i]=2，则在对应编码器阶段使用步幅卷积进行下采样。
        strides=(1, 1, 1, 1, 1),
        # 对应编码器阶段的卷积块中的卷积层数。
        enc_num_convs=(2, 2, 2, 2, 2),
        # 对应解码器阶段的卷积块中的卷积层数。
        dec_num_convs=(2, 2, 2, 2),
        # 对应编码器阶段后是否使用MaxPool对特征图进行下采样，若对应阶段使用步幅卷积（strides[i]=2），它永远不会使用 MaxPool下采样，甚至下采样[i-1]=True。
        downsamples=(True, True, True, True),
        # 编码器中每个阶段的膨胀率。
        enc_dilations=(1, 1, 1, 1, 1),
        # 解码器中每个阶段的膨胀率。
        dec_dilations=(1, 1, 1, 1),
        # 是否使用检查点。 使用检查点会在减慢训练速度的同时节省一些内存。
        with_cp=False,
        # 为卷积层配置字典。
        conv_cfg=None,
        # 配置规范化层的字典。
        norm_cfg=norm_cfg,
        # 配置 ConvModule 中激活层的字典。
        act_cfg=dict(type='ReLU'),
        # 解码器中上采样模块的上采样配置。
        upsample_cfg=dict(type='InterpConv'),
        # 是否将 norm 层设置为 eval 模式，即冻结运行状态（mean 和 var）。注意：仅对 Batch Norm 及其变体的影响。
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))
