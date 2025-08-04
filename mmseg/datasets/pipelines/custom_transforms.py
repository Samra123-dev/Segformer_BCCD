import numpy as np
from mmseg.datasets.builder import PIPELINES
from typing import Dict, Any
from albumentations import ColorJitter as AlbColorJitter


@PIPELINES.register_module()
class ConvertLabels:
    """
    Convert multi-class mask to binary mask:
    - Background (0) remains 0
    - All other labels (>0) become 1 (WBC)
    This is useful when original dataset has multiple WBC subtypes.
    """

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        label = results['gt_semantic_seg']
        label = (label > 0).astype(np.uint8)
        results['gt_semantic_seg'] = label
        return results


@PIPELINES.register_module()
class CutOut:
    """
    CutOut augmentation:
    Randomly masks out square patches in the image to simulate occlusions.
    Args:
        n_holes (int): Number of patches per image.
        cutout_ratio (float): Size of each patch relative to image.
        prob (float): Probability of applying CutOut.
    """

    def __init__(self, n_holes=8, cutout_ratio=0.1, prob=0.5):
        self.n_holes = n_holes
        self.cutout_ratio = cutout_ratio
        self.prob = prob

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]
        cutout_size_h = int(h * self.cutout_ratio)
        cutout_size_w = int(w * self.cutout_ratio)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - cutout_size_h // 2, 0, h)
            y2 = np.clip(y + cutout_size_h // 2, 0, h)
            x1 = np.clip(x - cutout_size_w // 2, 0, w)
            x2 = np.clip(x + cutout_size_w // 2, 0, w)

            img[y1:y2, x1:x2, :] = 0  # Zero out the patch

        results['img'] = img
        return results
@PIPELINES.register_module()  # Register in MMSegmentation's pipeline
class ColorJitter(AlbColorJitter):
    """Wrapper for Albumentations' ColorJitter to work with MMSegmentation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        # Apply Albumentations' ColorJitter to image
        augmented = super().__call__(image=results['img'])
        results['img'] = augmented['image']
        return results