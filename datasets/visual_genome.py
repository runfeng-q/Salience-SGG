import os
import os.path
import shutil
import json

import torch
import torch.utils.data
import numpy as np
from pycocotools import mask as coco_mask

import datasets.transforms as T
from datasets.abstract_datamodule import AbstractDataSet, AbstractDataModule
from datasets.coco import CocoDetectionDataset
from util.box_ops import box_cxcywh_to_xyxy



class VGDetection(CocoDetectionDataset):
    def __init__(self, img_folder, ann_file, return_masks=False, debug=False, stage='val'):
        super(VGDetection, self).__init__(img_folder, ann_file, return_masks, stage)
        self.debug = debug
    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(VGDetection, self).__getitem__(idx)
        target["labels"] -= 1  # remove 'no_relation' category
        #target["labels"] = torch.ones_like(target["labels"], dtype=torch.int64)
        return img, target

    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.ids)


class VGDataset(VGDetection):
    def __init__(
        self, img_folder, ann_file, return_masks=False, debug=False, stage='val', num_queries=200):
        super(VGDataset, self).__init__(img_folder, ann_file, return_masks, debug, stage)
        realtion_file=os.path.join('/'.join(ann_file.split('/')[:-1]),'rel.json')
        with open(realtion_file, "r") as f:
            rel = json.load(f)

        self.rel = rel[stage]
        self.rel_categories = rel["rel_categories"][1:]  # remove 'no_relation' category
        self.num_object_queries = num_queries
        self.stage=stage

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
                        #T.RandomSizeCrop(384, 600),
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
        # read in PIL image and target in COCO format
        img, target = super(CocoDetectionDataset, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        target["labels"] -= 1

        num_box = target["labels"].shape[0]

        rel_list = self.rel[str(int(image_id))]
        rel = np.array(rel_list)

        target["rel"] = self._get_rel_tensor(rel, num_box)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        rel = torch.zeros([self.num_object_queries, self.num_object_queries, 50])
        inds=target["rel"].shape
        rel[:inds[0],:inds[1], :]=target["rel"]
        target["rel"]=rel
        return img, target

    def _get_rel_tensor(self, rel_tensor, num_box):
        indices = rel_tensor.T
        indices[-1, :] -= 1  # remove 'no_relation' category
        #rel = torch.zeros([self.num_object_queries, self.num_object_queries, 50])
        rel = torch.zeros([num_box, num_box, 50])
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel
    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.ids)

class VGDataModule(AbstractDataModule):
    def __init__(
            self,
            config):
        super().__init__(config)
        self.dataset_name=config['dataset'].split('_')[0]
        self.task=config['dataset'].split('_')[-1]
        self.processed_dir = os.path.join(self.root, 'processed', self.dataset_name)
        self.check_table={
            'images': 108249,
            'vg': 9
        }
        self.annotations_path = os.path.join(self.processed_dir,'vg')
        self.data_dir = os.path.join(self.processed_dir,'images')

    def extract_data(self,data_path):
        #if raw data does not exist, download under https://github.com/yrcong/RelTR/blob/main/data/README.md
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_path=os.path.join(self.root, 'raw', 'VG')
        img_path=os.path.join(raw_path, 'VG_100K')
        annotation_path=os.path.join(raw_path, 'vg')
        shutil.copytree(img_path, self.data_dir)
        shutil.copytree(annotation_path, self.annotations_path)

    def setup(self, stage=None):
        if self.task=='DETECT':
            self.train_dataset =VGDetection(self.data_dir, f'{self.annotations_path}/train.json', stage='train')
            self.val_dataset = VGDetection(self.data_dir, f'{self.annotations_path}/val.json', stage='val')
            self.test_dataset=VGDetection(self.data_dir, f'{self.annotations_path}/test.json', stage='test')
        elif self.task=='SGG':
            self.train_dataset = VGDataset(self.data_dir, f'{self.annotations_path}/train.json', stage='train', num_queries=self.num_queries)
            self.val_dataset = VGDataset(self.data_dir, f'{self.annotations_path}/val.json', stage='val', num_queries=self.num_queries)
            self.test_dataset = VGDataset(self.data_dir, f'{self.annotations_path}/test.json', stage='test', num_queries=self.num_queries)
    @classmethod
    def collate_fn(cls, batch):
        imgs = []
        targets = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1])

        return {'imgs': imgs, 'targets': targets}
