from .builder import DATASETS
from .custom import CustomDataset
import os
from mmcv.parallel import DataContainer
import torch  # Required for binarization and tensor ops

@DATASETS.register_module()
class HematoDataset(CustomDataset):
    """Binary WBC segmentation: Background (0), WBC (1)"""

    CLASSES = ('Background', 'WBC')
    PALETTE = [
        [0, 0, 0],        # Background
        [255, 255, 255]   # WBC
    ]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='_mask.png',
            reduce_zero_label=False,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
        img_infos = []
        for filename in os.listdir(img_dir):
            if not filename.endswith(img_suffix):
                continue
            seg_filename = filename.replace(img_suffix, seg_map_suffix)
            img_path = os.path.join(img_dir, filename)
            seg_path = os.path.join(ann_dir, seg_filename)
            if not os.path.exists(seg_path):
                print(f"Mask not found: {seg_filename}")
                continue
            img_infos.append(dict(filename=filename, ann=dict(seg_map=seg_filename)))

        if not img_infos:
            raise RuntimeError("No image-mask pairs found.")
        print(f"Loaded {len(img_infos)} samples from {img_dir}")
        return img_infos

    def pre_pipeline(self, results):
        results['seg_fields'] = ['gt_semantic_seg']
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['reduce_zero_label'] = self.reduce_zero_label

    def prepare_train_img(self, idx):
        results = super().prepare_train_img(idx)
        seg = results['gt_semantic_seg'].data

        # Binarize: convert everything except background to WBC
        seg = (seg != 0).long()

        # Optional: print class distribution once
        if idx == 0:
            unique, counts = torch.unique(seg, return_counts=True)
            print(f" Class distribution in first sample: {dict(zip(unique.tolist(), counts.tolist()))}")

        results['gt_semantic_seg'] = DataContainer(seg, stack=True)
        return results

    def prepare_test_img(self, idx):
        return super().prepare_test_img(idx)
