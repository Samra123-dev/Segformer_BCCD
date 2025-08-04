_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/hemato.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# Model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=8,  # Updated from 7 to 8 classes
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            # Updated class weights for 8 classes
            class_weight=[0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Evaluation and runtime
evaluation = dict(interval=4000, metric='mIoU')
optimizer = dict(lr=6e-5)  # Only override learning rate

# Additional recommended settings for medical image segmentation
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
