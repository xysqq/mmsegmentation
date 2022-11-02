_base_ = [
    '../base/deeplabv3_unet_s5-d16.py', '../base/base_dicom_dataset.py',
    '../base/default_runtime.py', '../base/schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2),
             test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
evaluation = dict(metric='mDice')
