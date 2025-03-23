import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A

class FloorDataset(Dataset):
    """支持检测与分割的多任务数据集"""
    def __init__(self, img_dir, ann_dir, imgsz=640, augment=True):
        self.img_files = [f for f in Path(img_dir).glob('*.jpg')]
        self.ann_files = [Path(ann_dir)/f.stem.replace('image', 'mask') for f in self.img_files]
        
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(imgsz, imgsz),
            A.Normalize()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_files[index]))
        
        # 解析标注：检测框和分割掩码
        with open(self.ann_files[index].with_suffix('.txt')) as f:
            lines = f.readlines()
            boxes = [list(map(float, line.strip().split()[1:5])) for line in lines]
            classes = [int(line.strip().split()[0]) for line in lines]
        
        mask = np.load(self.ann_files[index].with_suffix('.npy'))
        
        # 数据增强
        transformed = self.transforms(image=img, bboxes=boxes, mask=mask)
        img = transformed['image'].transpose(2,0,1)
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        mask = torch.tensor(transformed['mask'], dtype=torch.long)
        
        return {
            'img': img,
            'bboxes': boxes,
            'cls': torch.tensor(classes),
            'mask': mask
        }