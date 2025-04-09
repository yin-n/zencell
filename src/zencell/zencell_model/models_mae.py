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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from zencell.zencell_model.util.pos_embed import (
    get_3d_sincos_pos_embed_from_grid,
)


class PatchEmbed3D(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=(2, 8, 8),
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )
        self.flatten = flatten
        self.in_chans = in_chans

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert self.img_size == (D, H, W)
        assert self.in_chans == C
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        ctx_size: Tuple[int, int, int],
        roi_size: Tuple[int, int, int],
        ctx_patch_size: Tuple[int, int, int],
        roi_patch_size: Tuple[int, int, int],
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        layer_scale=None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()

        assert all(
            x % y == 0 for x, y in zip(ctx_size, ctx_patch_size, strict=False)
        )
        assert all(
            x % y == 0 for x, y in zip(roi_size, ctx_patch_size, strict=False)
        )
        assert all(
            x % y == 0
            for x, y in zip(ctx_patch_size, roi_patch_size, strict=False)
        )

        # --------------------------------------------------------------------------
        # MAE encoder specifics
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

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    init_values=layer_scale,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.decoder_ctx_pos_embed = nn.Parameter(
            torch.zeros(1, ctx_num_patches, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding
        self.decoder_roi_pos_embed = nn.Parameter(
            torch.zeros(1, roi_num_patches, decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token_ctx = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )
        self.mask_token_roi = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    init_values=layer_scale,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        out_chans = in_chans
        self.decoder_pred_ctx = nn.Linear(
            decoder_embed_dim, math.prod(ctx_patch_size) * out_chans, bias=True
        )
        self.decoder_pred_roi = nn.Linear(
            decoder_embed_dim, math.prod(roi_patch_size) * out_chans, bias=True
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans

        self.initialize_weights()

        ctx_tok_id = torch.arange(ctx_num_patches).view(
            self.patch_embed_ctx.grid_size
        )
        ctx_fg_mask = torch.zeros_like(ctx_tok_id, dtype=torch.bool)
        roi_offset_3d = [
            (ctx_size - roi_size) // 2 // ctx_patch_size
            for ctx_size, roi_size, ctx_patch_size in zip(
                self.patch_embed_ctx.img_size,
                self.patch_embed_roi.img_size,
                self.patch_embed_ctx.patch_size,
                strict=False,
            )
        ]
        roi_slices_3d = [
            slice(st, st + sz // ctx_patch_size)
            for st, sz, ctx_patch_size in zip(
                roi_offset_3d,
                self.patch_embed_roi.img_size,
                self.patch_embed_ctx.patch_size,
                strict=False,
            )
        ]
        ctx_fg_mask[*roi_slices_3d] = 1

        self.ctx_fg_id = ctx_tok_id[ctx_fg_mask]
        self.ctx_bg_id = ctx_tok_id[~ctx_fg_mask]

        self.roi_fg_id = (
            torch.arange(roi_num_patches)
            .view(
                roi_size[0] // ctx_patch_size[0],
                ctx_patch_size[0] // roi_patch_size[0],
                roi_size[1] // ctx_patch_size[1],
                ctx_patch_size[1] // roi_patch_size[1],
                roi_size[2] // ctx_patch_size[2],
                ctx_patch_size[2] // roi_patch_size[2],
            )
            .permute(0, 2, 4, 1, 3, 5)
            .flatten(0, 2)
            .flatten(1)
        )

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        ctx_grid_d, ctx_grid_h, ctx_grid_w = [
            np.arange(grid_size) * patch_size + patch_size / 2
            for grid_size, patch_size in zip(
                self.patch_embed_ctx.grid_size,
                self.patch_embed_ctx.patch_size,
                strict=False,
            )
        ]
        roi_offset_3d = [
            (ctx_size - roi_size) // 2 // ctx_patch_size * ctx_patch_size
            for ctx_size, roi_size, ctx_patch_size in zip(
                self.patch_embed_ctx.img_size,
                self.patch_embed_roi.img_size,
                self.patch_embed_ctx.patch_size,
                strict=False,
            )
        ]
        roi_grid_d, roi_grid_h, roi_grid_w = [
            np.arange(grid_size) * patch_size + patch_size / 2 + roi_offset
            for grid_size, patch_size, roi_offset in zip(
                self.patch_embed_roi.grid_size,
                self.patch_embed_roi.patch_size,
                roi_offset_3d,
                strict=False,
            )
        ]
        ctx_grid = np.stack(
            np.meshgrid(ctx_grid_d, ctx_grid_h, ctx_grid_w), axis=0
        )
        roi_grid = np.stack(
            np.meshgrid(roi_grid_d, roi_grid_h, roi_grid_w), axis=0
        )
        ctx_pos_embed = get_3d_sincos_pos_embed_from_grid(
            self.ctx_pos_embed.shape[-1], ctx_grid
        )
        roi_pos_embed = get_3d_sincos_pos_embed_from_grid(
            self.roi_pos_embed.shape[-1], roi_grid
        )
        decoder_ctx_pos_embed = get_3d_sincos_pos_embed_from_grid(
            self.decoder_ctx_pos_embed.shape[-1], ctx_grid
        )
        decoder_roi_pos_embed = get_3d_sincos_pos_embed_from_grid(
            self.decoder_roi_pos_embed.shape[-1], roi_grid
        )

        self.ctx_pos_embed.data.copy_(
            torch.from_numpy(ctx_pos_embed).float().unsqueeze(0)
        )
        self.roi_pos_embed.data.copy_(
            torch.from_numpy(roi_pos_embed).float().unsqueeze(0)
        )
        self.decoder_ctx_pos_embed.data.copy_(
            torch.from_numpy(decoder_ctx_pos_embed).float().unsqueeze(0)
        )
        self.decoder_roi_pos_embed.data.copy_(
            torch.from_numpy(decoder_roi_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in [self.patch_embed_ctx, self.patch_embed_roi]:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        for param in [
            self.cls_token,
            self.mask_token_ctx,
            self.mask_token_roi,
        ]:
            nn.init.normal_(param, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, C, D, H, W)
        x: (N, L, patch_size**2 *3)
        """
        B, C, D, H, W = imgs.size()
        pD, pH, pW = patch_size
        return (
            imgs.view(B, C, D // pD, pD, H // pH, pH, W // pW, pW)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .flatten(1, 3)
            .flatten(2)
        )

    def unpatchify(self, x):
        raise NotImplementedError()
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

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

    def sync_random_masking(self, x_ctx, x_roi, mask_ratio):
        B, _, D = x_ctx.size()
        device = x_ctx.device

        def postprocess(x, ids_shuffle, len_keep):
            assert (
                ids_shuffle.min().item() == 0
                and ids_shuffle.max().item() == x.size(1) - 1
                and ids_shuffle.size(1) == x.size(1)
            )
            x_masked = torch.gather(
                x,
                dim=1,
                index=ids_shuffle[:, :len_keep, None].expand(-1, -1, D),
            )
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            mask = torch.ones([B, ids_shuffle.size(1)], device=device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return x_masked, mask, ids_restore

        self.ctx_fg_id = self.ctx_fg_id.to(device)
        self.ctx_bg_id = self.ctx_bg_id.to(device)
        self.roi_fg_id = self.roi_fg_id.to(device)
        ctx_fg_id = self.ctx_fg_id
        ctx_bg_id = self.ctx_bg_id
        roi_fg_id = self.roi_fg_id

        ids_shuffle_fg = torch.argsort(
            torch.rand([B, ctx_fg_id.size(0)], device=device), dim=1
        )
        ids_shuffle_bg = torch.argsort(
            torch.rand([B, ctx_bg_id.size(0)], device=device), dim=1
        )
        ids_shuffle_fg_ctx = ctx_fg_id[ids_shuffle_fg]
        ids_shuffle_fg_roi = roi_fg_id[ids_shuffle_fg].flatten(-2)
        ids_shuffle_bg_ctx = ctx_bg_id[ids_shuffle_bg]
        len_keep_fg = round(ctx_fg_id.size(0) * (1.0 - mask_ratio))
        len_keep_bg = round(ctx_bg_id.size(0) * (1.0 - mask_ratio))
        ids_shuffle_ctx = torch.cat(
            [
                ids_shuffle_fg_ctx[:, :len_keep_fg],
                ids_shuffle_bg_ctx[:, :len_keep_bg],
                ids_shuffle_fg_ctx[:, len_keep_fg:],
                ids_shuffle_bg_ctx[:, len_keep_bg:],
            ],
            dim=1,
        )
        x_ctx_masked, mask_ctx, ids_restore_ctx = postprocess(
            x_ctx,
            ids_shuffle_ctx,
            len_keep_fg + len_keep_bg,
        )

        len_keep_roi = len_keep_fg * roi_fg_id.size(-1)
        x_roi_masked, mask_roi, ids_restore_roi = postprocess(
            x_roi,
            ids_shuffle_fg_roi,
            len_keep_roi,
        )

        return (
            x_ctx_masked,
            x_roi_masked,
            mask_ctx,
            mask_roi,
            ids_restore_ctx,
            ids_restore_roi,
        )

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x_ctx = self.patch_embed_ctx(x) + self.ctx_pos_embed + self.ctx_embed
        x_roi = (
            self.patch_embed_roi(self.crop_roi(x))
            + self.roi_pos_embed
            + self.roi_embed
        )

        # masking: length -> length * mask_ratio
        x_ctx, x_roi, mask_ctx, mask_roi, ids_restore_ctx, ids_restore_roi = (
            self.sync_random_masking(x_ctx, x_roi, mask_ratio)
        )

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_ctx, x_roi), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x_cls, x_ctx, x_roi = x.tensor_split([1, 1 + x_ctx.size(1)], dim=1)

        return (
            x_cls,
            x_ctx,
            x_roi,
            mask_ctx,
            mask_roi,
            ids_restore_ctx,
            ids_restore_roi,
        )

    def forward_decoder(
        self, x_cls, x_ctx, x_roi, ids_restore_ctx, ids_restore_roi
    ):
        # embed tokens
        x_cls, x_ctx, x_roi = self.decoder_embed(
            torch.cat([x_cls, x_ctx, x_roi], dim=1)
        ).tensor_split([x_cls.size(1), x_cls.size(1) + x_ctx.size(1)], dim=1)
        mask_tokens_ctx = self.mask_token_ctx.repeat(
            x_ctx.size(0), ids_restore_ctx.size(1) - x_ctx.size(1), 1
        )
        mask_tokens_roi = self.mask_token_roi.repeat(
            x_roi.size(0), ids_restore_roi.size(1) - x_roi.size(1), 1
        )
        x_ctx = torch.cat([x_ctx, mask_tokens_ctx], dim=1)
        x_roi = torch.cat([x_roi, mask_tokens_roi], dim=1)
        x_ctx = torch.gather(
            x_ctx,
            dim=1,
            index=ids_restore_ctx.unsqueeze(-1).repeat(1, 1, x_ctx.size(2)),
        )
        x_roi = torch.gather(
            x_roi,
            dim=1,
            index=ids_restore_roi.unsqueeze(-1).repeat(1, 1, x_roi.size(2)),
        )
        x_ctx = x_ctx + self.decoder_ctx_pos_embed
        x_roi = x_roi + self.decoder_roi_pos_embed
        x = torch.cat([x_cls, x_ctx, x_roi], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_cls, x_ctx, x_roi = x.tensor_split(
            [x_cls.size(1), x_cls.size(1) + x_ctx.size(1)], dim=1
        )
        pred_ctx = self.decoder_pred_ctx(x_ctx)
        pred_roi = self.decoder_pred_roi(x_roi)

        return pred_ctx, pred_roi

    def forward_recon_loss(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        (
            latent_cls,
            latent_ctx,
            latent_roi,
            mask_ctx,
            mask_roi,
            ids_restore_ctx,
            ids_restore_roi,
        ) = self.forward_encoder(imgs, mask_ratio)
        pred_ctx, pred_roi = self.forward_decoder(
            latent_cls,
            latent_ctx,
            latent_roi,
            ids_restore_ctx,
            ids_restore_roi,
        )
        loss_recon_ctx = self.forward_recon_loss(
            self.patchify(imgs, self.patch_embed_ctx.patch_size),
            pred_ctx,
            mask_ctx,
        )
        loss_recon_roi = self.forward_recon_loss(
            self.patchify(
                self.crop_roi(imgs), self.patch_embed_roi.patch_size
            ),
            pred_roi,
            mask_roi,
        )

        return (
            loss_recon_ctx,
            loss_recon_roi,
            pred_ctx,
            pred_roi,
            mask_ctx,
            mask_roi,
        )


def mae_vit_large_patch2x8x8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=(2, 8, 8),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch1x8x8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=(1, 8, 8),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch2x8x8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=(2, 8, 8),
        in_chans=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_large_roipatch2x8x8_ctxpatch2x64x64_dep10_roi256_ctx1280_layerscale = (
    partial(
        MaskedAutoencoderViT,
        roi_patch_size=(2, 8, 8),
        ctx_patch_size=(2, 64, 64),
        roi_size=(10, 256, 256),
        ctx_size=(10, 1280, 1280),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        layer_scale=1e-5,
    )
)

mae_vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale = (
    partial(
        MaskedAutoencoderViT,
        roi_patch_size=(2, 8, 8),
        ctx_patch_size=(2, 64, 64),
        roi_size=(40, 256, 256),
        ctx_size=(40, 1280, 1280),
        in_chans=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        layer_scale=1e-5,
    )
)
