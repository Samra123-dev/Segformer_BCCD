import os.path as osp
import mmcv
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BCCDBinaryDataset(CustomDataset):
    CLASSES = ('background', 'bloodcell')  # Update as per your classes

    PALETTE = [[0, 0, 0], [255, 255, 255]]  # black background, white cell

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
    """
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
        # List all image files with given suffix
        img_list = mmcv.scandir(img_dir, img_suffix, recursive=True)
        data_infos = []
        for img in img_list:
            data_info = dict()
            data_info['filename'] = osp.join(img_dir, img)  # Use 'filename' key here
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                data_info['ann_path'] = osp.join(ann_dir, seg_map)
            data_infos.append(data_info)
        return data_infos
    """
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None):
        self.ann_dir = ann_dir
        img_list = mmcv.scandir(img_dir, img_suffix, recursive=True)
        data_infos = []

        for img in img_list:
            data_info = dict()
            data_info['filename'] = osp.join(img_dir, img)

            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                seg_map_path = osp.join(ann_dir, seg_map)

                if not osp.exists(seg_map_path):
                    print(f"[WARNING] Missing mask for: {img}")
                    continue

           
                data_info['ann_path'] = seg_map_path

          
                data_info['ann'] = {'seg_map': seg_map}

            else:
                print(f"[ERROR] ann_dir is None for image: {img}")
                continue

            data_infos.append(data_info)

        return data_infos

    def get_ann_info(self, idx):
        info = self.img_infos[idx]
        ann_path = info.get('ann_path', None)
        return dict(seg_map=ann_path)
