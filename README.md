Dataset link "https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask"

Training Script:python tools/train.py configs/segformer/B1/segformer.b1.512x512.bccd.160k.py  work_dirs/bccd

Testing Script:python tools/test.py configs/segformer/B1/segformer.b1.512x512.bccd.160k.py work_dirs/bccd/latest.pth

Inference Script:python tools/inference.py configs/path to this file local_configs/segformer/B1/segformer.b1.512x512.bccd.160k.py   work_dirs/bccd/latest.pth
