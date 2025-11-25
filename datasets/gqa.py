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

def load_info(dict_file):
    info = json.load(open(dict_file, 'r'))
    ind_to_classes = info['ind_to_classes']
    ind_to_predicates = info['ind_to_predicates']
    return ind_to_classes, ind_to_predicates

def load_graphs(data_json_file, split):
    data_info_all = json.load(open(data_json_file, 'r'))
    filenames = data_info_all['filenames_all']
    img_info = data_info_all['img_info_all']
    gt_boxes = data_info_all['gt_boxes_all']
    gt_classes = data_info_all['gt_classes_all']
    relationships = data_info_all['relationships_all']

    output_filenames = []
    output_img_info = []
    output_boxes = []
    output_classes = []
    output_relationships = []

    items = 0
    for filename, imginfo, gt_b, gt_c, gt_r in zip(filenames, img_info, gt_boxes, gt_classes, relationships):
        len_obj = len(gt_b)
        items += 1

        if split == 'val' or split == 'test':
            if items == 5580:
                continue

        if len(gt_r) > 0 and len_obj > 0:
            output_filenames.append(filename)
            output_img_info.append(imginfo)
            output_boxes.append(np.array(gt_b))
            output_classes.append(np.array(gt_c))
            output_relationships.append(np.array(gt_r))

    if split == 'val':
        output_filenames = output_filenames[:5000]
        output_img_info = output_img_info[:5000]
        output_boxes = output_boxes[:5000]
        output_classes = output_classes[:5000]
        output_relationships = output_relationships[:5000]

    return output_filenames, output_img_info, output_boxes, output_classes, output_relationships

class GQADetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, return_masks=False, debug=False, stage='val'):
        self.annotation_file=ann_file
        self.img_folder=img_folder
        data_folder='/'.join(img_folder.split("/")[:-1])
        self.cate_info_file = f"{data_folder}/GQA200/GQA_200_ID_Info.json"

        self.ind_to_classes, self.ind_to_predicates = load_info(self.cate_info_file)
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}
        self.split=stage
        self.debug = debug
        self._transforms = self.make_transform(stage)

        self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.annotation_file, self.split)

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img = Image.open(os.path.join(self.img_folder, self.filenames[idx])).convert("RGB")
        if img.size[0] != self.img_info[idx]['width'] or img.size[1] != self.img_info[idx]['height']:
            print('=' * 20, ' ERROR index ', str(idx), ' ', str(img.size), ' ', str(self.img_info[idx]['width']),
                  ' ', str(self.img_info[idx]['height']), ' ', '=' * 20)

        coco_target = self.convert_to_coco_format(idx)
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


    def convert_to_coco_format(self, index):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        boxes = self.gt_boxes[index]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor(self.gt_classes[index], dtype=torch.int64)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        annotation={
            "image_id": torch.as_tensor(index),
            "labels": classes-1,
            "boxes": boxes,
            "area": (boxes[:,3] - boxes[:,1] + 1) * (boxes[:,2] - boxes[:,0] + 1),
            "iscrowd": torch.zeros_like(classes),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        return annotation

    def __len__(self):
        if self.debug:
            return 5
        else:
            return len(self.filenames)


class GQADataset(GQADetection):
    def __init__(
        self, img_folder, ann_file, return_masks=False, debug=False, stage='val', num_queries=200, filter_duplicate_rels=True, filter_multiple_rels=False):
        super(GQADataset, self).__init__(img_folder, ann_file, return_masks, debug, stage)
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == "train"
        self.filter_multiple_rels = filter_multiple_rels and split == "train"
        self.num_object_queries = num_queries
        self.stage = stage
        self.rel_categories = self.ind_to_predicates[1:]

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
        img = Image.open(os.path.join(self.img_folder, self.filenames[idx])).convert("RGB")
        if img.size[0] != self.img_info[idx]['width'] or img.size[1] != self.img_info[idx]['height']:
            print('=' * 20, ' ERROR index ', str(idx), ' ', str(img.size), ' ', str(self.img_info[idx]['width']),
                  ' ', str(self.img_info[idx]['height']), ' ', '=' * 20)

        coco_target = self.convert_to_coco_format(idx)
        if self._transforms is not None:
            img, coco_target = self._transforms(img, coco_target)
        rel = torch.zeros([self.num_object_queries, self.num_object_queries, len(self.ind_to_predicates)-1])
        inds = coco_target["rel"].shape
        rel[:inds[0], :inds[1], :] = coco_target["rel"]
        coco_target["rel"] = rel

        return img, coco_target

    def convert_to_coco_format(self, index):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        boxes = self.gt_boxes[index]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor(self.gt_classes[index], dtype=torch.int64)
        num_box = boxes.shape[0]

        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]

        relation = self.relationships[index].copy()
        if self.filter_multiple_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        relation=self._get_rel_tensor(relation, num_box)
        relation=relation[keep][:,keep]
        annotation={
            "image_id": torch.as_tensor(index),
            "labels": classes-1,
            "boxes": boxes,
            "area": (boxes[:,3] - boxes[:,1] + 1) * (boxes[:,2] - boxes[:,0] + 1),
            "iscrowd": torch.zeros_like(classes),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "rel": relation,
        }
        return annotation

    def _get_rel_tensor(self, rel_tensor, num_box):
        indices = rel_tensor.T
        indices[-1, :] -= 1
        rel = torch.zeros([num_box, num_box, len(self.ind_to_predicates)-1])
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel

    def __len__(self):
        #if self.debug and self.split == "train":
        if self.debug:
            return 5
        else:
            return len(self.filenames)

class GQADataModule(AbstractDataModule):
    def __init__(
            self,
            config):
        super().__init__(config)
        self.dataset_name=config['dataset'].split('_')[0]
        self.task=config['dataset'].split('_')[-1]
        self.processed_dir = os.path.join(self.root, 'processed', self.dataset_name)
        self.check_table={
            'images': 148854,
            'GQA200': 3
        }
        self.annotations_path = os.path.join(self.processed_dir,'GQA200')
        self.data_dir = os.path.join(self.processed_dir,'images')

    def extract_data(self,data_path):
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_path=os.path.join(self.root, 'raw', 'GQA')
        img_path=os.path.join(raw_path, 'images')
        annotation_path=os.path.join(raw_path, 'GQA200')
        shutil.copytree(img_path, self.data_dir)
        shutil.copytree(annotation_path, self.annotations_path)

    def setup(self, stage=None):
        if self.task=='DETECT':
            self.train_dataset =GQADetection(self.data_dir, f'{self.annotations_path}/GQA_200_Train.json', stage='train')
            self.val_dataset = GQADetection(self.data_dir, f'{self.annotations_path}/GQA_200_Test.json', stage='val')
            self.test_dataset=GQADetection(self.data_dir, f'{self.annotations_path}/GQA_200_Test.json', stage='test')
        elif self.task=='SGG':
            self.train_dataset = GQADataset(self.data_dir, f'{self.annotations_path}/GQA_200_Train.json', stage='train', num_queries=200)
            self.val_dataset = GQADataset(self.data_dir, f'{self.annotations_path}/GQA_200_Test.json', stage='val', num_queries=200)
            self.test_dataset = GQADataset(self.data_dir, f'{self.annotations_path}/GQA_200_Test.json', stage='test', num_queries=200)
    @classmethod
    def collate_fn(cls, batch):
        imgs = []
        targets = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(sample[1])

        return {'imgs': imgs, 'targets': targets}