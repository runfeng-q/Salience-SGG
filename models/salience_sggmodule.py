import torch
import lightning as L
import torchvision
import numpy as np
import seaborn as sns

from .salience_sgg import *
import util.misc as utils
from util import box_ops
from util.misc import ParserObject, is_dist_avail_and_initialized, get_rank, is_main_process
from util.box_ops import rescale_bboxes
from lib.pytorch_misc import argsort_desc
from evaluation import calculate_mR_from_evaluator_list
from evaluation import CocoEvaluator,  VGDetectEvaluator, BasicSceneGraphEvaluator


class SalienceSGGModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.args = ParserObject(config)
        self.model, self.criterion, self.postprocessors = build(self.args)
        self._device = self.model.parameters().__next__().device
        torch.set_float32_matmul_precision('high')

    def _load_weights(self):
        if self.args.resume:
            checkpoint = torch.load(self.args.resume['ckpt'], map_location='cpu')
            pre_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['model']
            # For the case that the checkpoint was not saved with lightning but torch
            for key in list(pre_weights.keys()):
                if key.startswith('model'):
                    pre_weights[key[6:]] = pre_weights[key]
                    pre_weights.pop(key)
                elif key.startswith('criterion'):
                    pre_weights.pop(key)
            # Check if the shape of class embedding correponds to number of classes
            weight_shape = pre_weights['class_embed.0.weight'].shape
            if weight_shape[0] != self.args.num_classes:
                for key in CLASS_EMBEDDINGS_KEYS:
                    pre_weights.pop(key)

            if 'layers' in self.args.resume:
                for key in list(pre_weights.keys()):
                    if not any([key.startswith(layer) for layer in self.args.resume['layers']]):
                        pre_weights.pop(key)

            missing_keys, unexpected_keys = self.model.load_state_dict(pre_weights, strict=False)
            if 'transformer.decoder.layers.0.self_attn.q_proj.weight' in missing_keys and 'transformer.decoder.layers.0.self_attn.in_proj_weight' in unexpected_keys:
                for i in range(6):
                    for j, pro in enumerate(['q_proj', 'k_proj', 'v_proj']):
                        pre_weights[f'transformer.decoder.layers.{i}.self_attn.{pro}.weight'] = pre_weights[
                                                                                                    f'transformer.decoder.layers.{i}.self_attn.in_proj_weight'][
                                                                                                (j * 256):((
                                                                                                                       j + 1) * 256),
                                                                                                :]
                        pre_weights[f'transformer.decoder.layers.{i}.self_attn.{pro}.bias'] = pre_weights[
                                                                                                  f'transformer.decoder.layers.{i}.self_attn.in_proj_bias'][
                                                                                              (j * 256):((j + 1) * 256)]
                    pre_weights.pop(f'transformer.decoder.layers.{i}.self_attn.in_proj_weight')
                    pre_weights.pop(f'transformer.decoder.layers.{i}.self_attn.in_proj_bias')
                missing_keys, unexpected_keys = self.model.load_state_dict(pre_weights, strict=False)
            print('Model loaded')
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
        else:
            print('No available checkpoint')
    def setup(self, stage=None):
        if is_dist_avail_and_initialized():
            if is_main_process():
                self._load_weights()
        else:
            self._load_weights()

        for key, value in self.model.named_parameters():
            if 'add_add_add_relation' in key:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples = batch['imgs']
        targets = batch['targets']
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict  # A dictionary preserving the weight for every loss
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log_dict({'loss': loss}, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss}

    def on_validation_epoch_start(self) -> None:
        self.val_output_list = []

    def validation_step(self, batch, batch_idx):
        samples = [i.to(self._device) for i in batch['imgs']]
        labels = [
            {k: v.to(self._device) for k, v in t.items()} for t in batch["targets"]
        ]  # these are in DETR format, resized + normalized
        # forward pass
        outputs = self(samples)
        loss_dict = self.criterion(outputs, labels)
        self.val_output_list.append({'_loss_confidence': loss_dict['loss_confidence'], '_loss_relation': loss_dict['loss_rel']})

    def on_validation_epoch_end(self) -> None:
        recall={}
        avg_confidence_loss = torch.stack([x['_loss_confidence'] for x in self.val_output_list]).mean()
        avg_rel_loss = torch.stack([x['_loss_relation'] for x in self.val_output_list]).mean()
        recall['loss_confidence'] = avg_confidence_loss
        recall['loss_rel'] = avg_rel_loss

    def on_test_start(self) -> None:
        self.model.eval()

    def test_step(self, batch, batch_idx):
        samples = [i.to(self._device) for i in batch['imgs']]
        labels = [
            {k: v.to(self._device) for k, v in t.items()} for t in batch["targets"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = self(samples)
        pred_rels = outputs['pred_rel']
        rel_confideces=outputs['pred_confidence'][-1].sigmoid()

        for j, target in enumerate(labels):
            orig_size = target["orig_size"]
            target_labels = target["labels"]  # [num_objs]
            target_boxes = target["boxes"]  # [num_objs, 4]
            target_rel = target["rel"].nonzero()  # [num_rels, 3(s, o, p)]

            pred_boxes = outputs["pred_boxes"][j]
            pred_logits=outputs["pred_logits"][j].sigmoid()


            if 'gqa_single_sgg_evaluator' in self.evaluator:
                obj_scores, pred_classes = torch.max(
                    pred_logits[:, :self.args.num_classes], -1
                )
                pred_rel = pred_rels[j] - 0.2 * self.model.transformer.decoder.rel_dist.log().to(
                    pred_rels.device
                )
                pred_rel = pred_rel.sigmoid()
            else:
                obj_scores, pred_classes = torch.max(
                    pred_logits.softmax(-1)[:, :self.args.num_classes], -1
                )
                pred_rel = pred_rels[j].sigmoid()

            pred_rel = torch.clamp(pred_rel, 0.0, 1.0)

            rel_confidece = torch.clamp(rel_confideces[j], 0.0, 1.0)

            pred_rel[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
            , :] = 0.0  # prevent self-connection

            rel_confidece[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
            ] = 0.0  # prevent self-connection

            pred_rel_score, pred_rel_cls = pred_rel.reshape(-1, pred_rel.shape[-1]).max(-1)
            pred_rel=pred_rel

            sub_ob_scores = torch.outer(obj_scores, obj_scores)
            sub_ob_scores=sub_ob_scores*rel_confidece
            sub_ob_scores[
                torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
            ] = 0.0  # prevent self-connection

            if 'oi_sgg_evaluator' in self.evaluator:
                gt_entry = {
                    "gt_relations": target_rel.cpu().clone().numpy(),
                    "gt_boxes": rescale_bboxes(target_boxes.cpu(),
                                               torch.flip(orig_size, dims=[0])).cpu().clone().numpy(),
                    "gt_classes": target_labels.cpu().clone().numpy(),
                    'image_id': target["image_id"],
                }

                obj_scores, pred_classes = torch.max(
                    pred_logits, -1
                )
                sbj_obj_inds = torch.cartesian_prod(
                    torch.arange(pred_logits.shape[0]), torch.arange(pred_logits.shape[0])
                )
                pred_rel=pred_rel*rel_confidece.unsqueeze(-1)
                pred_scores = (
                    pred_rel.cpu().clone().numpy().reshape(-1, pred_rel.size(-1))
                )  # (num_obj * num_obj, num_rel_classes)
                pred_entry = {
                        "pred_boxes": rescale_bboxes(
                            pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                        )
                        .clone()
                        .numpy(),
                        "pred_classes": pred_classes.cpu().clone().numpy(),
                        "obj_scores": obj_scores.cpu().clone().numpy(),
                        "sbj_obj_inds": sbj_obj_inds,
                        "pred_scores": pred_scores,
                    }
                self.evaluator['oi_sgg_evaluator'].update(
                    gt_entry, pred_entry
                )
                continue

            demo_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            keep = torchvision.ops.batched_nms(demo_boxes, obj_scores, pred_classes, 0.5)
            keep_classes = pred_classes[keep]
            ious = torchvision.ops.box_iou(demo_boxes, demo_boxes[keep])
            iou_assignments = torch.zeros_like(pred_classes)
            #
            #
            for class_id in torch.unique(keep_classes):
                curr_indices = torch.where(pred_classes == class_id)[0]
                curr_keep_indices = torch.where(keep_classes == class_id)[0]
                curr_ious = ious[curr_indices][:, curr_keep_indices]
                curr_iou_assignment = curr_keep_indices[curr_ious.argmax(-1)]
                iou_assignments[curr_indices] = curr_iou_assignment

            iou_assignments_1 = iou_assignments.unsqueeze(-1).repeat(1, 200).unsqueeze(-1)
            iou_assignments_2 = iou_assignments.unsqueeze(0).repeat(200, 1).unsqueeze(-1)

            demo_assing = torch.arange(200).to('cuda')
            demo_1 = demo_assing.unsqueeze(-1).repeat(1, 200).unsqueeze(-1)
            demo_2 = demo_assing.unsqueeze(0).repeat(200, 1).unsqueeze(-1)
            demo = torch.cat((demo_1, demo_2), dim=-1).reshape(-1, 2)
            relationships = torch.cat((demo, pred_rel_cls.unsqueeze(-1)), dim=-1)

            rel_pair_idx = torch.cat((iou_assignments_1, iou_assignments_2), dim=-1).reshape(-1, 2)
            triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores).reshape(-1)
            pred_rel_label = torch.max(pred_rel, dim=-1)[1].reshape(-1)

            _, sorting_idx = torch.sort(triplet_scores, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx, :]
            result_pred_rel_scores = pred_rel.reshape(-1, pred_rel.shape[-1])[sorting_idx, :]
            rel_labels = pred_rel_label[sorting_idx]
            demo = demo[sorting_idx]

            triplets = torch.cat((rel_pair_idx, rel_labels.unsqueeze(-1)), -1)

            keep_triplet = torch.zeros_like(rel_labels)

            unique, idx, counts = torch.unique(triplets, dim=0, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to('cuda'), cum_sum[:-1]))
            first_indices = ind_sorted[cum_sum]
            keep_triplet[first_indices] = 1

            result_pred_rel_scores = result_pred_rel_scores[keep_triplet == 1]  # (#rel, #rel_class)
            demo = demo[keep_triplet == 1]

            connections = torch.zeros((target["rel"].shape[0], target["rel"].shape[1], 1)).to(target_rel.device)
            for re in target_rel:
                connections[re[0], re[1], 0] = 1

            if target_rel.nonzero().shape[0]==0:
                continue
            gt_entry = {
                "gt_relations": target_rel.cpu().clone().numpy(),
                "gt_boxes": rescale_bboxes(target_boxes.cpu(), torch.flip(orig_size, dims=[0])).cpu().clone().numpy(),
                "gt_classes": target_labels.cpu().clone().numpy(),
                "gt_connections": connections.nonzero().cpu().clone().numpy(),
            }


            if 'single_sgg_evaluator' in self.evaluator:
                pred_entry = {
                    "pred_boxes": rescale_bboxes(
                        pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                    )
                    .clone()
                    .numpy(),
                    "pred_classes": pred_classes.cpu().clone().numpy(),
                    "obj_scores": obj_scores.cpu().clone().numpy(),
                    "pred_rel_inds": demo[:100, :].cpu().clone().numpy(),
                    "rel_scores": result_pred_rel_scores[:100, :].cpu().clone().numpy(),
                }
                res, _, _ = self.evaluator['single_sgg_evaluator']["sgdet"].evaluate_scene_graph_entry(
                    gt_entry, pred_entry
                )

                for pred_id, _, evaluator_rel in self.evaluator['single_sgg_evaluator_list']:
                    gt_entry_rel = gt_entry.copy()
                    mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                    gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                    if gt_entry_rel["gt_relations"].shape[0] == 0:
                        continue
                    evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                        gt_entry_rel, pred_entry
                   )
            if 'gqa_single_sgg_evaluator' in self.evaluator:
                pred_entry = {
                    "pred_boxes": rescale_bboxes(
                        pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                    )
                    .clone()
                    .numpy(),
                    "pred_classes": pred_classes.cpu().clone().numpy(),
                    "obj_scores": obj_scores.cpu().clone().numpy(),
                    "pred_rel_inds": demo[:100, :].cpu().clone().numpy(),
                    "rel_scores": result_pred_rel_scores[:100, :].cpu().clone().numpy(),
                }
                res, _, _ = self.evaluator['gqa_single_sgg_evaluator']["sgdet"].evaluate_scene_graph_entry(
                    gt_entry, pred_entry
                )

                for pred_id, _, evaluator_rel in self.evaluator['gqa_single_sgg_evaluator_list']:
                    gt_entry_rel = gt_entry.copy()
                    mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                    gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                    if gt_entry_rel["gt_relations"].shape[0] == 0:
                        continue
                    evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                        gt_entry_rel, pred_entry
                   )
        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0
        )
        results = self.postprocessors['bbox'](
            outputs, orig_target_sizes
        )  # convert outputs of model to COCO api

        res = {
            target["image_id"].item(): output for target, output in zip(labels, results)
        }
        if 'detect_sgg' in self.evaluator:
            self.evaluator['detect_sgg'].update(res)
        if 'oi_detect_sgg' in self.evaluator:
            self.evaluator['oi_detect_sgg'].update(res)

    @staticmethod
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                "lr": float(self.args.lr),
            }
        ]
        if self.args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=float(self.args.lr), momentum=0.9,
                                        weight_decay=float(self.args.weight_decay))
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=float(self.args.lr),
                                          weight_decay=float(self.args.weight_decay))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)

        return [optimizer], [lr_scheduler]





