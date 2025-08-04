model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',  # local path to mit_b1 pretrained weights
    backbone=dict(
        type='mit_b1',
        style='pytorch'
    ),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.2, 1.0],
                reduction='mean',
                ignore_index=255
            ),
            dict(
                type='DiceLoss',
                loss_weight=3.0,
            )
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
