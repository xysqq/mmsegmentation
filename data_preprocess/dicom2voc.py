import os
import platform
import time
from glob import glob
from pathlib import Path

import mmcv
import numpy as np
from tqdm import tqdm

from utils.utils_dicom import ct_dicom2img, load_rtstruct_file, get_patient_id_from_dicom
from utils.utils_image_matching import load_dicom_volumes, match_ct_and_mri
from utils.utils_image_registration import register_images


def get_target_masks(roi_names, rtstruct):
    target_masks = {}
    for roi_name in roi_names:
        if roi_name in category_map.keys():
            if category_map[roi_name] in target_masks.keys():
                target_masks[category_map[roi_name]] += rtstruct.get_roi_mask_by_name(roi_name)
            target_masks[category_map[roi_name]] = rtstruct.get_roi_mask_by_name(roi_name)
    return target_masks


def dicom2voc(ct_dicom_dir, ct_structure, patient_id, out_dir):
    rtstruct, roi_names, body_masks = load_rtstruct_file(ct_dicom_dir, ct_structure)
    target_masks = get_target_masks(roi_names, rtstruct)

    for idx, file_dateset in enumerate(rtstruct.series_data):
        ct_dicom_path = file_dateset.filename
        slice_location = file_dateset.SliceLocation

        if slice_location < -150 or slice_location > 108:
            continue

        height, width = rtstruct.series_data[0].Rows, rtstruct.series_data[0].Columns

        image = ct_dicom2img(ct_dicom_path, body_masks[..., idx])
        masks = [target_masks[category][..., idx].astype(int) if category in target_masks.keys() else np.zeros(
            (height, width)) for category in category_list]

        class_seg = np.zeros((height, width))
        for i, mask in enumerate(masks):
            class_seg = np.where(
                class_seg > 0, class_seg, mask * (i + 1))
        mmcv.imwrite(image, f"{out_dir}/{patient_id}_{Path(ct_dicom_path).name}.jpg")
        mmcv.imwrite(class_seg, f"{out_dir}/{patient_id}_{Path(ct_dicom_path).name}.png")


def mutil_dicom2voc(ct_dicom_dir, ct_structure, mri_dicom_dirs, patient_id, out_dir):
    rtstruct, roi_names, body_masks = load_rtstruct_file(ct_dicom_dir, ct_structure)
    target_masks = get_target_masks(roi_names, rtstruct)

    ct_volumes, mri_volumes, ct_dicom_path_list, start_index = load_dicom_volumes(ct_dicom_dir,
                                                                                  mri_dicom_dirs,
                                                                                  ct_structure)
    result_volumes, _ = register_images(ct_volumes, mri_volumes)

    for idx, file_dateset in enumerate(rtstruct.series_data):
        ct_dicom_path = file_dateset.filename
        slice_location = file_dateset.SliceLocation

        if slice_location < -150 or slice_location > 108 or ct_dicom_path not in ct_dicom_path_list:
            continue

        height, width = rtstruct.series_data[0].Rows, rtstruct.series_data[0].Columns

        image = np.stack([ct_volumes[..., idx - start_index], result_volumes[0][..., idx - start_index],
                          result_volumes[1][..., idx - start_index]],
                         axis=2)
        masks = [target_masks[category][..., idx].astype(int) if category in target_masks.keys() else np.zeros(
            (height, width)) for category in category_list]

        class_seg = np.zeros((height, width))
        for i, mask in enumerate(masks):
            class_seg = np.where(
                class_seg > 0, class_seg, mask * (i + 1))
        mmcv.imwrite(image, f"{out_dir}/{patient_id}_{Path(ct_dicom_path).name}.jpg")
        mmcv.imwrite(class_seg, f"{out_dir}/{patient_id}_{Path(ct_dicom_path).name}.png")


if __name__ == '__main__':
    if platform.system() == 'Windows':
        data_dir = "D:/NPC"
    else:
        data_dir = "/home/xys/Data"

    images_ct_dir = rf"{data_dir}/images_ct_voc"
    images_fused_dir = rf"{data_dir}/images_fused_voc"
    os.makedirs(f"{images_ct_dir}/train", exist_ok=True)
    os.makedirs(f"{images_ct_dir}/val", exist_ok=True)
    os.makedirs(f"{images_fused_dir}/train", exist_ok=True)
    os.makedirs(f"{images_fused_dir}/val", exist_ok=True)

    ct_dicom_dirs = glob(rf"{data_dir}/npc/*/首次CT/*/*/CT")

    # 读取映射表
    category_map = {
        ' '.join(x.strip().split()[:-1]): x.strip().split(' ')[-1].strip()
        for x in open(rf"{data_dir}/label_target.txt", encoding='utf-8').readlines()
    }
    category_list = ['GTV', 'GTVnd']

    patient_id_list = list(set(ct_dicom_dir.split('/')[5] for ct_dicom_dir in ct_dicom_dirs))
    n = len(patient_id_list)
    split_index = int(0.8 * n)
    train_patient_id_list = patient_id_list[:split_index]

    start_time = time.time()
    for ct_dicom_dir in tqdm(ct_dicom_dirs):
        patient_id = get_patient_id_from_dicom(ct_dicom_dir)
        ct_dir = glob(rf"{data_dir}/npc/{patient_id}/首次CT/*/*/CT")[0]
        structure = glob(
            rf"{data_dir}/npc/{patient_id}/首次CT/*/*/RTSTRUCT/*")[0]
        mri_T1_dir = rf"{data_dir}/npc/{patient_id}/MR/S2010"
        mri_T2_dir = rf"{data_dir}/npc/{patient_id}/MR/S3010"
        # 找到CT图像与MRI图像之间的对应关系
        ct_mri_matching_dic = match_ct_and_mri(
            str(ct_dicom_dir), str(mri_T1_dir))
        if patient_id in train_patient_id_list:
            dicom2voc(ct_dicom_dir, structure, patient_id, f"{images_ct_dir}/train")
            mutil_dicom2voc(ct_dicom_dir, structure, [mri_T1_dir, mri_T2_dir], patient_id,
                            f"{images_fused_dir}/train")
        else:
            dicom2voc(ct_dicom_dir, structure, patient_id, f"{images_ct_dir}/val")
            mutil_dicom2voc(ct_dicom_dir, structure, [mri_T1_dir, mri_T2_dir], patient_id,
                            f"{images_fused_dir}/val")
    end_time = time.time()
    print(f"处理数据共用了{(end_time - start_time) / 60:.2f}分钟")
