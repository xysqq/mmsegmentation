_base_ = [
    '../base/fcn_unet_s5-d16.py', '../base/base_dicom_dataset.py',
    '../base/default_runtime.py', '../base/schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=22), auxiliary_head=dict(num_classes=22),
             test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
evaluation = dict(metric='mDice')

load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_64x64_40k_drive/fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth"
