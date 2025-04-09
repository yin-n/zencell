# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from zencell.zencell_model.models_mae import PatchEmbed3D


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        ctx_size: Tuple[int, int, int],
        roi_size: Tuple[int, int, int],
        ctx_patch_size: Tuple[int, int, int],
        roi_patch_size: Tuple[int, int, int],
        in_chans: int = 2,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Any = nn.LayerNorm,
        out_chans: int = 3,
        layer_scale: Optional[float] = None,
        qk_norm: float = False,
    ) -> None:
        super().__init__()
        self.patch_embed_ctx = PatchEmbed3D(
            ctx_size, ctx_patch_size, in_chans, embed_dim
        )
        self.patch_embed_roi = PatchEmbed3D(
            roi_size, roi_patch_size, in_chans, embed_dim
        )
        ctx_num_patches = self.patch_embed_ctx.num_patches
        roi_num_patches = self.patch_embed_roi.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ctx_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.roi_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ctx_pos_embed = nn.Parameter(
            torch.zeros(1, ctx_num_patches, embed_dim), requires_grad=False
        )
        self.roi_pos_embed = nn.Parameter(
            torch.zeros(1, roi_num_patches, embed_dim), requires_grad=False
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    init_values=layer_scale,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, math.prod(roi_patch_size) * out_chans, bias=True
        )
        self.out_chans = out_chans

    def crop_roi(self, imgs):
        roi_offset_3d = [
            (ctx_size - roi_size) // 2 // ctx_patch_size * ctx_patch_size
            for ctx_size, roi_size, ctx_patch_size in zip(
                self.patch_embed_ctx.img_size,
                self.patch_embed_roi.img_size,
                self.patch_embed_ctx.patch_size,
                strict=False,
            )
        ]
        roi_slices_3d = [
            slice(st, st + sz)
            for st, sz in zip(
                roi_offset_3d, self.patch_embed_roi.img_size, strict=False
            )
        ]
        return imgs[:, :, *roi_slices_3d]

    def forward(self, x):
        B = x.shape[0]
        x_ctx = self.patch_embed_ctx(x) + self.ctx_pos_embed + self.ctx_embed
        x_roi = (
            self.patch_embed_roi(self.crop_roi(x))
            + self.roi_pos_embed
            + self.roi_embed
        )

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x_ctx, x_roi), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x_cls, x_ctx, x_roi = x.tensor_split([1, 1 + x_ctx.size(1)], dim=1)

        x = x_roi
        x = self.norm(x)

        x = self.head(x)
        D, H, W = self.patch_embed_roi.img_size
        pD, pH, pW = self.patch_embed_roi.patch_size
        x = x.view(B, D // pD, H // pH, W // pW, pD, pH, pW, self.out_chans)
        x = (
            x.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .view(B, self.out_chans, D, H, W)
        )

        return x


vit_large_roipatch2x8x8_ctxpatch2x64x64_dep10_roi256_ctx1280_layerscale = (
    partial(
        VisionTransformer,
        roi_patch_size=(2, 8, 8),
        ctx_patch_size=(2, 64, 64),
        roi_size=(10, 256, 256),
        ctx_size=(10, 1280, 1280),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        layer_scale=1e-5,
    )
)

vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale = (
    partial(
        VisionTransformer,
        roi_patch_size=(2, 8, 8),
        ctx_patch_size=(2, 64, 64),
        roi_size=(40, 256, 256),
        ctx_size=(40, 1280, 1280),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        layer_scale=1e-5,
    )
)
