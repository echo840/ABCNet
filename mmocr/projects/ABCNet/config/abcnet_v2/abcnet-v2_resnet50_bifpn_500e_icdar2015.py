_base_ = [
    '_base_abcnet-v2_resnet50_bifpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_500e.py',
]

# dataset settings
icdar2015_textspotting_train = _base_.icdar2015_textspotting_train
icdar2015_textspotting_train.pipeline = _base_.train_pipeline
icdar2015_textspotting_test = _base_.icdar2015_textspotting_test
icdar2015_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)

load_from = '/home/user/lz/ABCNet/mmocr/projects/ABCNet/model/abcnet-v2_resnet50_bifpn_500e_icdar2015-5e4cc7ed.pth'  # noqa
#load_from = '/home/user/lz/ABCNet/mmocr/projects/ABCNet/model/ic15_pretrained.pth'

find_unused_parameters = True