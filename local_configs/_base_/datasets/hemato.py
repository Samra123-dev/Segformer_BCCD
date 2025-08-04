dataset_type = 'HematoDataset'
data_root = '/media/iml/cv-lab/Datasets_B_cells/Hemato_Data'

classes = ('Background', 'WBC')
palette = [
    [0, 0, 0],       # Background - black
    [128, 0, 0],     # WBC - dark red
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ConvertLabels'),  # Fixed this line
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ConvertLabels'),  # Fixed this line
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/Train/images',
        ann_dir=f'{data_root}/Train/masks',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/Validation/images',
        ann_dir=f'{data_root}/Validation/masks',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/Validation/images',
        ann_dir=f'{data_root}/Validation/masks',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        test_mode=True,
    )
)
