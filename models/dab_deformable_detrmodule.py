import torch
import lightning as L

from .dab_deformable_detr import *
import util.misc as utils
from util.box_ops import rescale_bboxes
from util.misc import ParserObject, is_dist_avail_and_initialized, get_rank,is_main_process

CLASS_EMBEDDINGS_KEYS = {
    'class_embed.0.weight',
    'class_embed.0.bias',
    'class_embed.1.weight',
    'class_embed.1.bias',
    'class_embed.2.weight',
    'class_embed.2.bias',
    'class_embed.3.weight',
    'class_embed.3.bias',
    'class_embed.4.weight',
    'class_embed.4.bias',
    'class_embed.5.weight',
    'class_embed.5.bias',
}

class DABDeformableDetrModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.args = ParserObject(config)
        self.model, self.criterion, self.postprocessors=build(self.args)
        self._device = self.model.parameters().__next__().device
        self.visual_threshold=self.args.visual_threshold
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
            #Check if the shape of class embedding correponds to number of classes
            weight_shape=pre_weights['class_embed.0.weight'].shape
            if weight_shape[0]!=self.args.num_classes:
                for key in CLASS_EMBEDDINGS_KEYS:
                    pre_weights.pop(key)
                    
            if 'layers' in self.args.resume:
                for key in list(pre_weights.keys()):
                    if not any([key.startswith(layer) for layer in self.args.resume['layers']]):
                        pre_weights.pop(key)

            missing_keys, unexpected_keys=self.model.load_state_dict(pre_weights, strict=False)
            if 'transformer.decoder.layers.0.self_attn.q_proj.weight' in missing_keys and 'transformer.decoder.layers.0.self_attn.in_proj_weight' in unexpected_keys:
                for i in range(6):
                    for j, pro in enumerate(['q_proj','k_proj','v_proj']):
                        pre_weights[f'transformer.decoder.layers.{i}.self_attn.{pro}.weight']=pre_weights[f'transformer.decoder.layers.{i}.self_attn.in_proj_weight'][(j*256):((j+1)*256),:]
                        pre_weights[f'transformer.decoder.layers.{i}.self_attn.{pro}.bias'] = pre_weights[f'transformer.decoder.layers.{i}.self_attn.in_proj_bias'][(j*256):((j+1)*256)]
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

    def setup(self,stage=None):
        if is_dist_avail_and_initialized():
            if is_main_process():
                self._load_weights()
        else:
            self._load_weights()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples=batch['imgs']
        targets=batch['targets']
        outputs=self(samples)
        loss_dict=self.criterion(outputs, targets)
        weight_dict=self.criterion.weight_dict                #A dictionary preserving the weight for every loss
        loss=sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log_dict({'loss': loss}, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def validation_step(self, batch, batch_idx):
        samples = batch['imgs']
        targets = batch['targets']
        outputs=self(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict  # A dictionary preserving the weight for every loss
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.val_output_list.append({'_loss': loss, '_ce_loss': loss_dict['loss_ce'],'_loss_bbox': loss_dict['loss_bbox']})
        #return {'_loss': loss, '_ce_loss': loss_dict['loss_ce'],'_loss_bbox': loss_dict['loss_bbox']}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['_loss'] for x in self.val_output_list]).mean()
        avg_ce_loss = torch.stack([x['_ce_loss'] for x in self.val_output_list]).mean()
        avg_loss_bbox = torch.stack([x['_loss_bbox'] for x in self.val_output_list]).mean()
        self.log_dict({'val_loss': avg_loss, 'loss_ce': avg_ce_loss, 'loss_bbox': avg_loss_bbox}, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # get the inputs
        samples= [i.to(self._device) for i in batch['imgs']]
        labels = [
            {k: v.to(self._device) for k, v in t.items()} for t in batch["targets"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs=self(samples)

        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0
        )
        results = self.postprocessors['bbox'](
            outputs, orig_target_sizes
        )  # convert outputs of model to COCO api

        res = {
            target["image_id"].item(): output for target, output in zip(labels, results)
        }
        if 'gqa_detect_sgg' in self.evaluator:
            for j, label in enumerate(labels):
                pred_boxes = outputs["pred_boxes"][j]
                pred_logits = outputs["pred_logits"][j].sigmoid()

                obj_scores, pred_classes = torch.max(
                    pred_logits, -1
                )
                pred_entry = {
                    "pred_boxes": rescale_bboxes(pred_boxes.cpu(),
                                               torch.flip(orig_target_sizes[j], dims=[0])).cpu().clone().numpy(),
                    "pred_classes": pred_classes.cpu().clone().numpy(),
                    "obj_scores": obj_scores.cpu().clone().numpy(),
                }
                gt_entry = {
                    "gt_boxes": rescale_bboxes(label['boxes'].cpu(),
                                               torch.flip(orig_target_sizes[j], dims=[0])).cpu().clone().numpy(),
                    "gt_classes": label['labels'].cpu().clone().numpy(),
                }
                self.evaluator['gqa_detect_sgg'].update(gt_entry,pred_entry)

        for k_eval, evaluator in self.evaluator.items():
            if k_eval=='detect_sgg':
                evaluator.update(res)
            elif k_eval=='oi_detect_sgg':
                evaluator.update(res)

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
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": float(self.args.lr_backbone),
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




