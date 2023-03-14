# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DicomDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = (
        'background', 'BODY', 'BrainStem', 'CTV', 'CTVnd', 'Chiasm', 'Cochlea',
        'Eye', 'GTV', 'GTVnd', 'Hippocampus', 'Lens', 'Mandible', 'Mastoid',
        'OpticNerve', 'Parotid', 'PharynxConst', 'SpinalCord', 'Submandibular',
        'TMjoint', 'TemporalLobe', 'Thyroid')

    PALETTE = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128], [64, 64, 64]
    ]

    def __init__(self, split, **kwargs):
        super(DicomDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
