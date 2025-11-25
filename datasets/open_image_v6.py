import os
import os.path
import tqdm
import shutil
import subprocess
import pickle
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import torch
import torch.utils.data
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from datasets.abstract_datamodule import AbstractDataSet, AbstractDataModule
from util.box_ops import box_cxcywh_to_xyxy

def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, "r"))
    ind_to_predicates_cate = info["rel"]
    ind_to_entites_cate = info["obj"]

    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (
        ind_to_entites_cate,
        ind_to_predicates_cate,
        entites_cate_to_ind,
        predicate_to_ind,
    )

class OIDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, return_masks=False, debug=False, stage='val'):
        self.annotation_file=ann_file
        self.img_folder=img_folder
        data_folder='/'.join(img_folder.split("/")[:-1])
        self.cate_info_file = f"{data_folder}/annotations/categories_dict.json"
        self.targets = json.load(open(self.annotation_file, "r"))
        (
            self.ind_to_classes,
            self.rel_categories,
            self.classes_to_ind,
            self.predicates_to_ind,
        ) = load_cate_info(self.cate_info_file)
        self.split=stage
        self.debug = debug
        self._transforms = self.make_transform(stage)
    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        target = self.targets[idx]
        img = Image.open(f"{self.img_folder}/{target['img_fn']}.jpg").convert("RGB")
        coco_target = self.convert_to_coco_format(idx, img.size)
        if self._transforms is not None:
            img, coco_target = self._transforms(img, coco_target)
        return img, coco_target

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

    def convert_to_coco_format(self, index, orig_size):
        w, h = orig_size
        target = self.targets[index]
        boxes =torch.as_tensor(target["bbox"], dtype=torch.float32).reshape(-1, 4)
        classes=torch.tensor(target["det_labels"], dtype=torch.int64)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        annotation={
            "image_id": torch.as_tensor(index),
            "labels": classes,
            "boxes": boxes,
            "area": (boxes[:,3] - boxes[:,1] + 1) * (boxes[:,2] - boxes[:,0] + 1),
            "iscrowd": torch.zeros_like(classes),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)])
        }
        return annotation

    def __len__(self):
        if self.debug:
            return 5
        else:
            return len(self.targets)


class OIDataset(OIDetection):
    def __init__(
        self, img_folder, ann_file, return_masks=False, debug=False, stage='val', num_queries=200, filter_duplicate_rels=True, filter_multiple_rels=False):
        super(OIDataset, self).__init__(img_folder, ann_file, return_masks, debug, stage)
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == "train"
        self.filter_multiple_rels = filter_multiple_rels and split == "train"
        self.remove_tail_classes = False
        self.num_object_queries = num_queries
        self.stage = stage
        self.categories = {
            i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))
        }
        if stage == "train":
            self.targets = [
                target
                for target in self.targets
                if len(target["bbox"]) <= self.num_object_queries
            ]
            if filter_duplicate_rels:
                # choose one relation between same subject and object
                assert self.stage == "train"
                for idx, target in enumerate(self.targets):
                    all_rel_sets = defaultdict(list)
                    for sbj, obj, rel in target["rel"]:
                        all_rel_sets[(sbj, obj, rel)].append(rel)
                    self.targets[idx]["rel"] = [
                        [k[0], k[1], v[0]] for k, v in all_rel_sets.items()
                    ]
        self.pre_compute_bbox = None

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
        target = self.targets[idx]
        img = Image.open(f"{self.img_folder}/{target['img_fn']}.jpg").convert("RGB")
        coco_target = self.convert_to_coco_format(idx, img.size)
        rel_list = target["rel"]
        if self.filter_multiple_rels:
            all_rel_sets = defaultdict(list)
            for sbj, obj, rel in rel_list:
                all_rel_sets[(sbj, obj)].append(rel)
            rel_list = [
                [k[0], k[1], np.random.choice(v)] for k, v in all_rel_sets.items()
            ]
        rel = np.array(rel_list)
        num_box = coco_target["labels"].shape[0]
        coco_target["rel"] = self._get_rel_tensor(rel, num_box)
        if self._transforms is not None:
            img, coco_target = self._transforms(img, coco_target)

        rel = torch.zeros([self.num_object_queries, self.num_object_queries, len(self.predicates_to_ind)])
        inds=coco_target["rel"].shape
        rel[:inds[0],:inds[1], :]=coco_target["rel"]
        coco_target["rel"] = rel
        return img, coco_target

    def _get_rel_tensor(self, rel_tensor, num_box):
        indices = rel_tensor.T
        rel = torch.zeros([num_box, num_box, len(self.predicates_to_ind)])
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel

    def __len__(self):
        #if self.debug and self.split == "train":
        if self.debug:
            return 5
        else:
            return len(self.targets)

class OIDataModule(AbstractDataModule):
    def __init__(
            self,
            config):
        super().__init__(config)
        self.dataset_name=config['dataset'].split('_')[0]
        self.task=config['dataset'].split('_')[-1]
        self.processed_dir = os.path.join(self.root, 'processed', self.dataset_name)
        self.check_table={
            'images': 133503,
            'annotations': 6
        }
        self.annotations_path = os.path.join(self.processed_dir,'annotations')
        self.data_dir = os.path.join(self.processed_dir,'images')

    def extract_data(self,data_path):
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_path=os.path.join(self.root, 'raw', 'OIv6', 'open-imagev6')
        img_path=os.path.join(raw_path, 'images')
        annotation_path=os.path.join(raw_path, 'annotations')
        shutil.copytree(img_path, self.data_dir)
        shutil.copytree(annotation_path, self.annotations_path)

    def setup(self, stage=None):
        if self.task=='DETECT':
            self.train_dataset =OIDetection(self.data_dir, f'{self.annotations_path}/vrd-train-anno.json', stage='train')
            self.val_dataset = OIDetection(self.data_dir, f'{self.annotations_path}/vrd-test-anno.json', stage='val')
            self.test_dataset=OIDetection(self.data_dir, f'{self.annotations_path}/vrd-test-anno.json', stage='test')
        elif self.task=='SGG':
            self.train_dataset = OIDataset(self.data_dir, f'{self.annotations_path}/vrd-train-anno.json', stage='train', num_queries=200)
            self.val_dataset = OIDataset(self.data_dir, f'{self.annotations_path}/vrd-test-anno.json', stage='val', num_queries=200)
            self.test_dataset = OIDataset(self.data_dir, f'{self.annotations_path}/vrd-test-anno.json', stage='test', num_queries=200)
    @classmethod
    def collate_fn(cls, batch):
        imgs = []
        targets = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1])

        return {'imgs': imgs, 'targets': targets}