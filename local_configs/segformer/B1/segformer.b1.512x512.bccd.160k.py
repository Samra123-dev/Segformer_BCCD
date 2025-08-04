_base_ = [
    '/media/iml/cv-lab/Segformer/SegFormer/local_configs/_base_/models/segformer.py',  # Adjusted path
    '/media/iml/cv-lab/Segformer/SegFormer/local_configs/_base_/datasets/bccd.py',     # Points to your bccd.py
    '/media/iml/cv-lab/Segformer/SegFormer/local_configs/_base_/default_runtime.py',
    '/media/iml/cv-lab/Segformer/SegFormer/local_configs/_base_/schedules/schedule_160k_adamw.py'
]
# Model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mit_b1',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mit_b1.pth')),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),  # ✅ Fix is here
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[0.2, 0.8],
            loss_weight=1.0))
)

# Dataset settings (if not in bccd.py)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# Runtime
evaluation = dict(interval=4000, metric='mIoU')