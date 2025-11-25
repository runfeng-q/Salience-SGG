import os
import os.path
import tqdm
import shutil
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from io import BytesIO
from pycocotools import mask as coco_mask
import torchvision

import datasets.transforms as T
from datasets.abstract_datamodule import AbstractDataSet, AbstractDataModule


class CocoDetectionDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, return_masks=False, stage='val'):
        super(CocoDetectionDataset, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._transforms = self.make_transform(stage)
    @staticmethod
    def make_transform(image_set):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

        if image_set == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

        if image_set == 'val' or image_set == 'test':
            return T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        raise ValueError(f'unknown {image_set}')

    def __getitem__(self, idx):
        img, target = super(CocoDetectionDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class COCODetectionDataModule(AbstractDataModule):
    def __init__(
            self,
            config):
        super().__init__(config)
        self.dataset_name=config['dataset']
        self.processed_dir = os.path.join(self.root, 'processed', self.dataset_name)
        self.check_table={
            'val2017': 5000,
            'annotations/instances_val2017.json': 19987840
        }
        self.annotations_path = os.path.join(self.processed_dir,'annotations/instances_val2017.json')
        self.data_dir = os.path.join(self.processed_dir,'val2017')

    def extract_data(self,data_path):
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_path=os.path.join(self.root, 'raw', 'COCO')
        img_path=os.path.join(raw_path, 'val2017')
        annotation_path=os.path.join(raw_path, 'annotations')
        shutil.copytree(img_path, self.data_dir)
        shutil.copytree(annotation_path, os.path.join(self.processed_dir, 'annotations'))

    def setup(self, stage=None):
        self.train_dataset = CocoDetectionDataset(self.data_dir, self.annotations_path, stage='train')
        self.val_dataset = CocoDetectionDataset(self.data_dir, self.annotations_path, stage='val')
        self.test_dataset=CocoDetectionDataset(self.data_dir, self.annotations_path, stage='test')

    @classmethod
    def collate_fn(cls, batch):
        imgs = []
        targets = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1])

        return {'imgs': imgs, 'targets': targets}