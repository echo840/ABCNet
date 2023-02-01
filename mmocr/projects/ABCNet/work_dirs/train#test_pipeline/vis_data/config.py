file_client_args = dict(backend='disk')
num_classes = 1
strides = [8, 16, 32, 64, 128]
bbox_coder = dict(type='mmdet.DistancePointBBoxCoder')
with_bezier = True
norm_on_bbox = True
use_sigmoid_cls = True
dictionary = dict(
    type='Dictionary',
    dict_file=
    '/home/user/lz/ABCNet/mmocr/projects/ABCNet/config/abcnet_v2/../../dicts/abcnet.txt',
    with_start=False,
    with_end=False,
    same_start_end=False,
    with_padding=True,
    with_unknown=True)
model = dict(
    type='ABCNet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[1, 1, 1],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        norm_cfg=dict(type='BN'),
        num_outs=6,
        relu_before_extra_convs=True),
    det_head=dict(
        type='ABCNetDetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        use_sigmoid_cls=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        use_scale=False,
        with_bezier=True,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='conv_cls',
                std=0.01,
                bias=-4.59511985013459)),
        module_loss=dict(
            type='ABCNetDetModuleLoss',
            num_classes=1,
            strides=[8, 16, 32, 64, 128],
            center_sampling=True,
            center_sample_radius=1.5,
            bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
            norm_on_bbox=True,
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        postprocessor=dict(
            type='ABCNetDetPostprocessor',
            use_sigmoid_cls=True,
            strides=[8, 16, 32, 64, 128],
            bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
            with_bezier=True,
            test_cfg=dict(
                nms_pre=1000,
                nms=dict(type='nms', iou_threshold=0.4),
                score_thr=0.3))),
    roi_head=dict(
        type='RecRoIHead',
        neck=dict(type='CoordinateHead'),
        roi_extractor=dict(
            type='BezierRoIExtractor',
            roi_layer=dict(
                type='BezierAlign', output_size=(16, 64), sampling_ratio=1.0),
            out_channels=256,
            featmap_strides=[4, 8, 16]),
        rec_head=dict(
            type='ABCNetRec',
            backbone=dict(type='ABCNetRecBackbone'),
            encoder=dict(type='ABCNetRecEncoder'),
            decoder=dict(
                type='ABCNetRecDecoder',
                dictionary=dict(
                    type='Dictionary',
                    dict_file=
                    '/home/user/lz/ABCNet/mmocr/projects/ABCNet/config/abcnet_v2/../../dicts/abcnet.txt',
                    with_start=False,
                    with_end=False,
                    same_start_end=False,
                    with_padding=True,
                    with_unknown=True),
                postprocessor=dict(
                    type='AttentionPostprocessor',
                    ignore_chars=['padding', 'unknown']),
                module_loss=dict(
                    type='CEModuleLoss',
                    ignore_first_char=False,
                    ignore_char=-1,
                    reduction='mean'),
                max_seq_len=25))),
    postprocessor=dict(
        type='ABCNetPostprocessor',
        rescale_fields=['polygons', 'bboxes', 'beziers']))
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2000, 4000), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2000, 4000), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
icdar2015_textspotting_data_root = 'data/icdar2015'
icdar2015_textspotting_train = dict(
    type='OCRDataset',
    data_root='data/icdar2015',
    ann_file='textspotting_train.json',
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(
            type='Resize',
            scale=(2000, 4000),
            keep_ratio=True,
            backend='pillow'),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True,
            with_text=True),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ])
icdar2015_textspotting_test = dict(
    type='OCRDataset',
    data_root='data/icdar2015',
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(
            type='Resize',
            scale=(2000, 4000),
            keep_ratio=True,
            backend='pillow'),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True,
            with_text=True),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ])
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = '/home/user/lz/ABCNet/mmocr/projects/ABCNet/model/abcnet-v2_resnet50_bifpn_500e_icdar2015-5e4cc7ed.pth'
resume = False
val_evaluator = [dict(type='E2EHmeanIOUMetric'), dict(type='HmeanIOUMetric')]
test_evaluator = [dict(type='E2EHmeanIOUMetric'), dict(type='HmeanIOUMetric')]
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextSpottingLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(type='value', clip_value=1))
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=500, val_interval=1, val_begin=0)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', end=1000, start_factor=0.001, by_epoch=False)
]
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textspotting_train.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(
                type='Resize',
                scale=(2000, 4000),
                keep_ratio=True,
                backend='pillow'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True,
                with_text=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textspotting_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(
                type='Resize',
                scale=(2000, 4000),
                keep_ratio=True,
                backend='pillow'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True,
                with_text=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textspotting_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(
                type='Resize',
                scale=(2000, 4000),
                keep_ratio=True,
                backend='pillow'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True,
                with_text=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)
find_unused_parameters = True
launcher = 'pytorch'
work_dir = 'work_dirs/'
