norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
        out_indices=(3,7,11),
        norm_eval=False,
    ),
    decode_head=dict(
        type='TBFormerHead',
        in_channels=768,
        channels=768,
        num_classes=2,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        convnormcfg=norm_cfg,
        num_levels=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    finetune=False,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
)

img_norm_cfg = dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]
data = dict(test=dict(pipeline=test_pipeline))