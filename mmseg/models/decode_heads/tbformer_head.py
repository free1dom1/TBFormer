# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmcv.runner import ModuleList
from mmseg.ops import Upsample
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1).contiguous()
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1).contiguous()).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 2
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        feat_fusion = feat_p

        return feat_fusion


@HEADS.register_module()
class TBFormerHead(BaseDecodeHead):
    """
    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            num_levels=1,
            use_DA=True,
            fpn=True,
            upsamp=False,
            convnormcfg=dict(type='BN'),
            convactcfg=dict(type='ReLU'),
            **kwargs,
    ):
        super(TBFormerHead, self).__init__(
            in_channels=in_channels, **kwargs)

        self.num_levels = num_levels
        self.use_DA = use_DA
        if self.use_DA:
            self.DAs = ModuleList()
            for _ in range(self.num_levels):
                self.DAs.append(_DAHead(1536, **kwargs))

        self.fpn = fpn
        if self.fpn:
            self.addconv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            self.addnorm = nn.BatchNorm2d(in_channels)
            self.addactivate = nn.ReLU()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.upsamp = upsamp
        if self.upsamp:
            self.upsp_convs = nn.Sequential(
                Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
                Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
                ConvModule(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=1, padding=1,
                           norm_cfg=convnormcfg, act_cfg=convactcfg),
                Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
                Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners))

        self.dec_proj = nn.Linear(in_channels, embed_dims)
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        if self.upsamp:
            self.convup_norm = build_norm_layer(
                norm_cfg, embed_dims, postfix=2)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=3)[1]

        self.init_std = init_std

        delattr(self, 'conv_seg')

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        inputs = [inputs[i] for i in range(self.num_levels)]

        if self.use_DA:
            for i, DA in enumerate(self.DAs):
                inputs[i] = DA(inputs[i])

        if self.fpn:
            for i in range(self.num_levels - 1, 0, -1):
                inputs[i - 1] = inputs[i - 1] + inputs[i]
            inputs[0] = self.addconv(inputs[0])
            inputs[0] = self.addnorm(inputs[0])
            inputs[0] = self.addactivate(inputs[0])
            x = inputs[0]
        else:
            x = inputs[-1]

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)

        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = x[:, :-self.num_classes].contiguous()
        cls_seg_feat = x[:, -self.num_classes:].contiguous()

        if self.upsamp:
            patches = patches.permute(0, 2, 1).contiguous().view(b, -1, h, w)
            patches = self.upsp_convs(patches)
            b, c, h, w = patches.shape
            patches = patches.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            patches = self.convup_norm(patches)
            cls_seg_feat = self.classes_proj(cls_seg_feat)
        else:
            patches = self.patch_proj(patches)
            cls_seg_feat = self.classes_proj(cls_seg_feat)

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2).contiguous()
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks

