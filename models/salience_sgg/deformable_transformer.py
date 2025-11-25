# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torchvision
import torch.nn.functional as F
import functools
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid, Conv2dNormActivation
from models.ops.modules import MSDeformAttn
from util import box_ops

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False, prior_static=None, num_queries=200, num_rel_labels=50, num_class=150, use_fre_bias=True, salience_layer=1,cascadic=False, glove_embed=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                    prior_static=prior_static,num_queries=num_queries, num_rel_labels=num_rel_labels, num_class=num_class, use_fre_bias=use_fre_bias, salience_layer=salience_layer,cascadic=cascadic,glove_embed=glove_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        #dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            # grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            #                                 torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                                            indexing="ij")
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # import ipdb; ipdb.set_trace()
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid()
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
                # bs, num_quires, 2
            init_reference_out = reference_points

        # decoder
        # import ipdb; ipdb.set_trace()
        hs, inter_references, outputs_class, outputs_coord, relations,confidence= self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_pos=query_embed if not self.use_dab else None,
                                            src_padding_mask=mask_flatten)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, outputs_class, outputs_coord, relations, confidence
        return hs, init_reference_out, inter_references_out, None, None, outputs_class, outputs_coord, relations, confidence

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            #                               torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing="ij")
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, att_masks=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=att_masks)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2, attention_weights, sampling_locations = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, attention_weights, sampling_locations

class DeformableTransformerSelfAttLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),attn_mask=mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerCrossAttLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, key, query_pos, mask=None,key_pos=None):
        # self attention
        q =self.with_pos_embed(tgt, query_pos)
        if key_pos==None:
            k=self.with_pos_embed(key, query_pos)
        else:
            k=self.with_pos_embed(key, key_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), key.transpose(0, 1),attn_mask=mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False,
                 d_model=256, high_dim_query_update=False, no_sine_embed=False, prior_static=None,
                 num_queries=200, num_rel_labels=50, num_class=150, use_fre_bias=True, salience_layer=1, cascadic=False, glove_embed=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        self.num_queries=num_queries
        self.num_rel_labels=num_rel_labels
        self.num_class=num_class
        self.salience_layer=salience_layer
        self.use_fre_bias=use_fre_bias
        self.cascadic=cascadic
        print(f'******use_fre_bias={self.use_fre_bias}*********')
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


        eps = float(1e-12)
        self.add_add_add_relation_object_rep = nn.Linear(d_model + 300 + d_model, 512)
        add_add_relation_embed_layer = Relation_Layer(512 * 2 + 64, 2048, self.num_rel_labels, 3, use_norm=True)
        nn.init.constant_(add_add_relation_embed_layer.layers[-1].weight.data, 0)
        nn.init.constant_(add_add_relation_embed_layer.layers[-1].bias.data, 0)
        self.add_add_add_relation_embed = add_add_relation_embed_layer
        self.add_add_add_relation_box_embed = MLP(4, d_model, d_model, 3)
        self.add_add_add_relation_dropout_rel = nn.Dropout(p=0.15)

        self.add_add_add_relation_to_subject = nn.Linear(d_model + self.num_class, d_model)
        self.add_add_add_relation_to_object = nn.Linear(d_model + self.num_class, d_model)
        self.add_add_add_relation_att_map = Conv2dNormActivation(1,
                                                             8,
                                                             kernel_size=1,
                                                             inplace=True,
                                                             norm_layer=None,
                                                             activation_layer=nn.ReLU)
        self.add_add_add_relation_rel_spe_att_map=nn.Sequential(MLP(self.num_rel_labels,200,8, 3),
                                                            nn.ReLU(),
                                                            )
        add_add_relation_sub_to_obj = DeformableTransformerCrossAttLayer(d_model=d_model, d_ffn=2048)
        add_add_relation_obj_to_sub = DeformableTransformerCrossAttLayer(d_model=d_model, d_ffn=2048)
        add_add_relation_selfatt_subject = DeformableTransformerSelfAttLayer(d_model=d_model, d_ffn=2048)
        add_add_relation_selfatt_object = DeformableTransformerSelfAttLayer(d_model=d_model, d_ffn=2048)
        add_add_relation_confidence_subject_layer = nn.Linear(d_model, d_model)
        add_add_relation_confidence_object_layer = nn.Linear(d_model, d_model)

        self.add_add_add_relation_sub_to_obj = _get_clones(add_add_relation_sub_to_obj, salience_layer)
        self.add_add_add_relation_obj_to_sub = _get_clones(add_add_relation_obj_to_sub,salience_layer)
        self.add_add_add_relation_selfatt_subject = _get_clones(add_add_relation_selfatt_subject, salience_layer)
        self.add_add_add_relation_selfatt_object = _get_clones(add_add_relation_selfatt_object,salience_layer)
        self.add_add_add_relation_confidence_subject_layer = _get_clones(add_add_relation_confidence_subject_layer, salience_layer)
        self.add_add_add_relation_confidence_object_layer = _get_clones(add_add_relation_confidence_object_layer, salience_layer)
        if self.cascadic:
            self.add_add_add_relation_query_scale = MLP(d_model, d_model, d_model, 2)
            self.add_add_add_relation_ref_point_head = MLP(d_model, d_model, d_model, 2)

        fg_matrix = torch.load(prior_static)[:,:,:num_rel_labels]
        rel_dist = torch.FloatTensor(
            (fg_matrix.sum(axis=0).sum(axis=0)) / (fg_matrix.sum() + eps)
        )

        triplet_dist = torch.FloatTensor(
            fg_matrix + eps / (fg_matrix.sum(2, keepdims=True) + eps)
        )

        triplet_dist = triplet_dist.log()
        self.rel_dist = nn.Parameter(rel_dist, requires_grad=False)
        self.triplet_dist = nn.Parameter(triplet_dist, requires_grad=False)
        del rel_dist, triplet_dist

        glove_embd = torch.load(glove_embed)
        self.glove_embd = nn.Parameter(glove_embd, requires_grad=False)

    def forward(self, tgt, reference_points, src, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        bs = src.shape[0]
        reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)
        intermediate = []
        intermediate_reference_points = []
        outputs_classes = []
        outputs_coords = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)
            #reverse_src_valid_ratios=src_valid_ratios.unsqueeze(1).unsqueeze(1).unsqueeze(-2) #N, Len_q, n_heads, n_levels, n_points, 2

            output, attention_weights, sampling_locations = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            # hack implementation for iterative bounding box refinement
            #samping_loc=sampling_locations/reverse_src_valid_ratios
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                outputs_class = self.class_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if lid==(self.num_layers-1):
                with torch.no_grad():
                    obj_cls = torch.argmax(outputs_classes[5].detach().sigmoid(), dim=-1)
                    obj_boxes = outputs_coords[5].detach()
                    box_matrix = box_rel_encoding(obj_boxes, obj_boxes)
                    overlap_emb = get_sine_pos_embed(box_matrix)
                    sematic_emb = torch.stack(
                        [
                            self.glove_embd[obj_cls[i]]
                            for i in range(len(obj_cls))
                        ],
                        dim=0,
                    )
                node_emb = self.add_add_add_relation_dropout_rel(output.detach())

                cls_emb = torch.cat((node_emb, sematic_emb, self.add_add_add_relation_box_embed(obj_boxes)), -1)
                cls_emb = self.add_add_add_relation_object_rep(cls_emb)
                cls_emb_object = cls_emb.clone()
                cls_emb_subject = cls_emb.clone()

                cls_emb_subject = (
                    cls_emb_subject
                    .unsqueeze(2)
                    .repeat(1, 1, self.num_queries, 1)
                )

                cls_emb_object = (
                    cls_emb_object
                    .unsqueeze(1)
                    .repeat(1, self.num_queries, 1, 1)
                )

                relation_feature = torch.cat((cls_emb_subject, cls_emb_object), -1)

                relation_feature = torch.cat((relation_feature, overlap_emb), -1)

                relation = self.add_add_add_relation_embed(relation_feature)
                if self.use_fre_bias:
                    relation += torch.stack(
                        [
                            self.triplet_dist[obj_cls[i]][:, obj_cls[i]]
                            for i in range(len(obj_cls))
                        ],
                        dim=0,
                    )
                else:
                    relation=relation

                with torch.no_grad():
                    if outputs_coords[5].shape[-1] == 4:
                        reference_points_input = outputs_coords[5][:, :, None] \
                                                 * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,
                                                   None]  # bs, nq, 4, 4
                    else:
                        assert outputs_coords[5].shape[-1] == 2
                        reference_points_input = outputs_coords[5][:, :, None] * src_valid_ratios[:, None]
                    if self.use_dab:
                        # import ipdb; ipdb.set_trace()
                        if self.no_sine_embed:
                            raw_query_pos = self.ref_point_head(reference_points_input)
                        else:
                            query_sine_embed = gen_sineembed_for_position(
                                reference_points_input[:, :, 0, :])  # bs, nq, 256*2
                            raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                        pos_scale = self.query_scale(output) if lid != 0 else 1
                        query_pos = pos_scale * raw_query_pos
                    if self.high_dim_query_update and lid != 0:
                        query_pos = query_pos + self.high_dim_query_proj(output)
                query_pos = query_pos.detach()
                relation_subject = torch.cat((node_emb, outputs_classes[5].sigmoid().detach()), -1)
                relation_object = torch.cat((node_emb, outputs_classes[5].sigmoid().detach()), -1)

                self_input_subject = self.add_add_add_relation_to_subject(relation_subject)
                self_input_object = self.add_add_add_relation_to_object(relation_object)

                iou_masks = []
                for v in range(bs):
                    demo_box2 = box_ops.box_cxcywh_to_xyxy(outputs_coords[5][v])
                    demo_iou = torchvision.ops.box_iou(demo_box2, demo_box2)
                    iou_masks.append(demo_iou.unsqueeze(-1))
                iou_masks = torch.stack(iou_masks).permute(0, 3, 1, 2).contiguous()
                iou_att_map = self.add_add_add_relation_att_map(iou_masks).contiguous().view(-1,
                                                                                         iou_masks.shape[
                                                                                             -2],
                                                                                         iou_masks.shape[
                                                                                             -1])
                rel_spe_map = self.add_add_add_relation_rel_spe_att_map(relation.sigmoid()).permute(0, 3, 1, 2).contiguous()
                rel_spe_map = rel_spe_map.view(-1, rel_spe_map.shape[-2], rel_spe_map.shape[-1])

                if not self.cascadic:
                    confidences=[]
                    for sil in range(self.salience_layer):
                        self_output_subject = self.add_add_add_relation_selfatt_subject[sil](self_input_subject, query_pos,
                                                                                    iou_att_map)
                        self_output_object = self.add_add_add_relation_selfatt_object[sil](self_input_object, query_pos, iou_att_map)

                        self_output_sub=self.add_add_add_relation_sub_to_obj[sil](self_output_subject, self_output_object, query_pos, mask=rel_spe_map)
                        self_output_obj = self.add_add_add_relation_obj_to_sub[sil](self_output_object, self_output_subject, query_pos, mask=rel_spe_map.transpose(-2,-1))

                        confidence_subject = self.add_add_add_relation_confidence_subject_layer[sil](self_output_sub)
                        confidence_object = self.add_add_add_relation_confidence_object_layer[sil](self_output_obj)
                        confidence = torch.bmm(confidence_subject, confidence_object.transpose(1, 2)) / (
                            self.d_model) ** 0.5
                        confidences.append(confidence)
                        self_input_subject = self_output_sub
                        self_input_object = self_output_obj
                else:
                    confidences = []
                    conf_sub=(torch.ones(bs,self.num_queries,self.num_queries)*float(1e-8)).to(relation.device).detach()
                    conf_obj=torch.eye(self.num_queries,self.num_queries).repeat(bs,1,1).to(relation.device).detach()

                    for sil in range(self.salience_layer):
                        query_salient_embed_sub = gen_sineembed_for_position(
                            conf_sub)  # bs, nq, 256*2
                        query_salient_embed_obj = gen_sineembed_for_position(
                            conf_obj)  # bs, nq, 256*2

                        raw_query_sub= self.add_add_add_relation_ref_point_head(query_salient_embed_sub)
                        raw_query_obj = self.add_add_add_relation_ref_point_head(query_salient_embed_obj)

                        pos_scale_sub = self.add_add_add_relation_query_scale(self_input_subject) if sil != 0 else 1
                        pos_scale_obj = self.add_add_add_relation_query_scale(self_input_object) if sil != 0 else 1

                        query_pos_sub = pos_scale_sub * raw_query_sub+query_pos
                        query_pos_obj = pos_scale_obj * raw_query_obj + query_pos

                        self_output_subject = self.add_add_add_relation_selfatt_subject[sil](self_input_subject, query_pos_sub,
                                                                                    iou_att_map)
                        self_output_object = self.add_add_add_relation_selfatt_object[sil](self_input_object, query_pos_obj,
                                                                                  iou_att_map)

                        self_output_sub = self.add_add_add_relation_sub_to_obj[sil](self_output_subject, self_output_object,
                                                                           query_pos_sub,
                                                                                    mask=rel_spe_map, key_pos=query_pos_obj
                                                                                    )
                        self_output_obj = self.add_add_add_relation_obj_to_sub[sil](self_output_object, self_output_subject,
                                                                           query_pos_obj,
                                                                           mask=rel_spe_map.transpose(-2, -1), key_pos=query_pos_sub
                                                                                    )

                        confidence_subject = self.add_add_add_relation_confidence_subject_layer[sil](self_output_sub)
                        confidence_object = self.add_add_add_relation_confidence_object_layer[sil](self_output_obj)


                        confidence = torch.bmm(confidence_subject, confidence_object.transpose(1, 2)) / (
                            self.d_model) ** 0.5
                        new_confidence=inverse_sigmoid(conf_sub)+confidence
                        confidences.append(new_confidence)
                        conf_sub=new_confidence.sigmoid().detach()
                        self_input_subject=self_output_sub
                        self_input_object = self_output_obj

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        confidences=torch.stack(confidences)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), outputs_classes, outputs_coords, relation, confidences
        return output, reference_points, outputs_classes, outputs_coords, relation, confidences

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed

@functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
    dim_t = temperature**(dim_t * 2 / num_pos_feats)
    return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)

def exchange_xy_fn(pos_res):
    index = torch.cat([
        torch.arange(1, -1, -1, device=pos_res.device),
        torch.arange(2, pos_res.shape[-2], device=pos_res.device),
    ])
    pos_res = torch.index_select(pos_res, -2, index)
    return pos_res

def get_sine_pos_embed(
    pos_tensor: Tensor,
    num_pos_feats: int = 16,
    temperature: int = 10000.,
    scale: float = 100.,
    exchange_xy: bool = False,
) -> Tensor:
    """Generate sine position embedding for a position tensor

    :param pos_tensor: shape as (..., 2*n).
    :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
    :param temperature: the temperature used for scaling the position embedding, defaults to 10000
    :param exchange_xy: exchange pos x and pos. For example,
        input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
    :return: position embedding with shape (None, n * num_pos_feats)
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    if exchange_xy:
        pos_res = exchange_xy_fn(pos_res)
    pos_res = pos_res.flatten(-2)
    return pos_res


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        use_dab=True,
        prior_static=args.prior_static,
        num_queries=args.num_queries,
        use_fre_bias=args.use_fre_bias,
        salience_layer=args.salience_layer,
        num_rel_labels=args.num_rel_labels,
        num_class=args.num_classes,
        cascadic=args.cascadic,
        glove_embed=args.glove_embed
    )


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Relation_Layer(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_norm=False, dropout=0.15):
        super().__init__()
        self.num_layers = num_layers
        self.use_norm = False
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if use_norm:
            self.use_norm = True
            self.layers.insert(0,nn.LayerNorm(input_dim))
            # self.layers.insert(2,nn.Dropout(p=dropout))
            # self.layers.insert(4, nn.Dropout(p=dropout))
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, retun_interm=False):
        if self.use_norm:
            x = self.layers[3](self.dropout2(F.relu(self.layers[2](self.dropout1(self.layers[1](self.layers[0](x)))))))
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                # if  retun_interm and i==self.num_layers-2 :
                #     interm_feats = x
            # if retun_interm:
            #     return x,interm_feats
            # else:
            #
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 200:
        pos = torch.cat((pos_y, pos_x), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

