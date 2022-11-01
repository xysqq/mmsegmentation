_base_ = [
    '../base/deeplabv3_unet_s5-d16.py', '../base/ct_voc12.py',
    '../base/default_runtime.py', '../base/schedule_40k.py'
]
model = dict(test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
evaluation = dict(metric='mDice')