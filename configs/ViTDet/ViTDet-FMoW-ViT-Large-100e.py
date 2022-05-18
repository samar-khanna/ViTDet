_base_ = [
    './xview_100e.py'
]

norm_cfg = dict(type='LN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
# Requires MMCV-full after  https://github.com/open-mmlab/mmcv/pull/1205.
head_norm_cfg = dict(type='LN', requires_grad=True)

pretrained = None  # noqa
# model settings
model = dict(
    type='RetinaNet',
    pretrained=pretrained,
    backbone=dict(
        type='VisionTransformerFMoW',
        img_size=416,
        num_classes=60,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        norm_cfg=norm_cfg,
        use_residual=False,
        num_outs=5),
    # rpn_head=dict(
    #     type='RPNHead',
    #     num_convs=2,
    #     norm_cfg=head_norm_cfg,
    #     in_channels=256,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=60,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # roi_head=dict(
    #     type='StandardRoIHead',
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]),
    #     bbox_head=dict(
    #         type='Shared4Conv1FCBBoxHead',
    #         conv_out_channels=256,
    #         norm_cfg=head_norm_cfg,
    #         in_channels=256,
    #         fc_out_channels=1024,
    #         roi_feat_size=7,
    #         num_classes=80,
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]),
    #         reg_class_agnostic=False,
    #         loss_cls=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            match_low_quality=True,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100),
    # train_cfg=dict(
    #     rpn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.7,
    #             neg_iou_thr=0.3,
    #             min_pos_iou=0.3,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=256,
    #             pos_fraction=0.5,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=False),
    #         allowed_border=-1,
    #         pos_weight=-1,
    #         debug=False),
    #     rpn_proposal=dict(
    #         nms_pre=2000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.5,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.5,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         mask_size=28,
    #         pos_weight=-1,
    #         debug=False)),
    # test_cfg=dict(
    #     rpn=dict(
    #         nms_pre=1000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100,
    #         mask_thr_binary=0.5))

)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.1,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                        num_layers=24,
                        layer_decay_rate=0.7,
                        custom_keys={
                            'bias': dict(decay_multi=0.),
                            'pos_embed': dict(decay_mult=0.),
                            'relative_position_bias_table': dict(decay_mult=0.),
                            'norm': dict(decay_mult=0.),
                            "rel_pos_h": dict(decay_mult=0.),
                            "rel_pos_w": dict(decay_mult=0.),
                            }
                            )
                 )
lr_config = dict(warmup_iters=250) # 16 * 1000 == 250 * 64
find_unused_parameters = True
