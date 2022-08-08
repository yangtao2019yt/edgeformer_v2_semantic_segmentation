# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_mv2.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='_MobileNetV2',
        width_mult=1.0,
        inverted_residual_setting=None, # use default settings
        round_nearest=8,
        block=None, # use default settings
        norm_layer=None, # use default settings
        out_indices=(1, 2, 4, 7),
        frozen_stages=-1,
    ),
    decode_head=dict(
        in_channels=[24, 32, 96, 1280],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=96,
        num_classes=150
    ),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 6})

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    # use_fp16=True,
    use_fp16=False,
)
