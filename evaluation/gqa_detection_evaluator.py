import os
import contextlib
import copy
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import math
from functools import reduce

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.pytorch_misc import argsort_desc, intersect_2d, ap_eval, prepare_mAP_dets

def _xyxy_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)

def eval_entites_detection(all_results, ind_to_classes):
    # create a Coco-like object that we can use to evaluate detection!
    anns = []
    result_str = ""

    for image_id, _result in enumerate(all_results):
        labels = _result["gt_class"].tolist()  # integer
        boxes = _result["gt_boxes"].tolist()  # xyxy
        for cls, box in zip(labels, boxes):
            anns.append(
                {
                    "area": (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0] + 1,
                        box[3] - box[1] + 1,
                    ],  # xywh
                    "category_id": cls,
                    "id": len(anns),
                    "image_id": image_id,
                    "iscrowd": 0,
                }
            )
    fauxcoco = COCO()
    fauxcoco.dataset = {
        "info": {"description": "use coco script for oi detection evaluation"},
        "images": [{"id": i} for i in range(len(all_results))],
        "categories": [
            {"supercategory": "person", "id": i-1, "name": name}
            for i, name in enumerate(ind_to_classes)
            if name != "__background__"
        ],
        "annotations": anns,
    }
    fauxcoco.createIndex()

    # format predictions to coco-like
    cocolike_predictions = []
    for image_id, _result in enumerate(all_results):
        box = _result["pred_boxes"]
        label = _result["pred_class"]
        score = _result["pred_cls_scores"]
        box = [_xyxy_to_xywh(_box) for _box in box]  # xywh
        image_id = np.asarray([image_id] * len(box))
        cocolike_predictions.append(np.column_stack((image_id, box, score, label)))
        # logger.info(cocolike_predictions)
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)

    res = fauxcoco.loadRes(cocolike_predictions)
    coco_eval = COCOeval(fauxcoco, res, "bbox")
    coco_eval.params.imgIds = list(range(len(all_results)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_res = COCOResults("bbox")
    coco_res.update(coco_eval)
    mAp = coco_eval.stats[1]

    def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
        p = coco_eval.params

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        if maxDets == -1:
            max_range_i = np.argmax(p.maxDets)
            mind = [
                max_range_i,
            ]
        else:
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if eval_type == "precision":
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif eval_type == "recall":
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        else:
            raise ValueError("Invalid eval metrics")
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return p.maxDets[mind[-1]], mean_s

    coco_res_to_save = {}
    for key, value in coco_res.results.items():
        for evl_name, eval_val in value.items():
            coco_res_to_save[f"{key}/{evl_name}"] = eval_val
    print(coco_res_to_save)

    result_str += "Detection evaluation mAp=%.4f\n" % mAp
    result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, "recall")
    result_str += "=" * 100 + "\n"
    avg_metrics = mAp
    print(result_str)
    return coco_res_to_save


class GQAEvaluator:
    def __init__(self, ind_to_classes):
        self.ind_to_classes = ind_to_classes
        self.all_result = []

    def update(
        self,
        gt_entry,
        pred_entry,
    ):
        gt_boxes = gt_entry["gt_boxes"]
        gt_class = gt_entry["gt_classes"]

        result_dict={}
        result_dict["gt_boxes"]=gt_boxes
        result_dict["gt_class"]=gt_class


        pred_boxes = pred_entry["pred_boxes"]
        pred_class = pred_entry["pred_classes"]
        pred_score = pred_entry["obj_scores"]
        result_dict["pred_boxes"]=pred_boxes
        result_dict["pred_class"]=pred_class
        result_dict["pred_cls_scores"]=pred_score
        for key, value in result_dict.items():
            result_dict[key] = np.array(value)
        self.all_result.append(result_dict)

    def aggregate_metrics(self):
        log_dict = {}
        log_dict.update(eval_entites_detection(self.all_result, self.ind_to_classes))
        print(log_dict)
        return log_dict