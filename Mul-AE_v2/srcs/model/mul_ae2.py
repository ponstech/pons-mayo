# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from random import randint
# from cv2 import repeat
import numpy as np


import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block

from srcs.utils.pos_embed import get_2d_sincos_pos_embed, interpolate_pos_embed
from srcs.utils import get_logger
from einops import rearrange


class ChannelMaskedAEViT(timm.models.vision_transformer.VisionTransformer):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool=False, num_classes=2, drop_path_rate=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
        )
        self.pos_drop = nn.Dropout(p=drop_path_rate)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(int(decoder_depth))])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # MAE decoder 2
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed2 = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks2 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth//2)])

        self.decoder_norm2 = norm_layer(decoder_embed_dim)
        self.decoder_pred2 = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder 3
        self.decoder_embed3 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token3 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed3 = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks3 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth//2)])

        self.decoder_norm3 = norm_layer(decoder_embed_dim)
        self.decoder_pred3 = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) # decoder to patch


        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed2 = get_2d_sincos_pos_embed(self.decoder_pos_embed2.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed2.data.copy_(torch.from_numpy(decoder_pos_embed2).float().unsqueeze(0))

        decoder_pos_embed3 = get_2d_sincos_pos_embed(self.decoder_pos_embed3.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed3.data.copy_(torch.from_numpy(decoder_pos_embed3).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token2, std=.02)
        torch.nn.init.normal_(self.mask_token3, std=.02)

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

    def patchify(self, x):
        """
        imgs: (N, 4, H, W)
        x: (N, 4 , L, patch_size**2)
        """
        N, C, H, W = x.shape
        p = self.patch_embed.patch_size[0]
        assert x.shape[2] == x.shape[3] and x.shape[2] % p == 0

        x = rearrange(x, 'n c (h p) (w q) -> n c (h w) (p q)', p=p, q=p)
        
        return x

    def unpatchify(self, x):
        """
        x: (N, C , L, patch_size**2)
        img_depth: (N, C, H, W)
        """
        N, C, L, D = x.shape
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[2]**.5)
        assert h * w == x.shape[2]
        
        x = rearrange(x, 'n c (h w) (p q) -> n c (h p) (w q)', h=h, p=p)
        
        return x

    
    def random_channel_mixing(self, x, channels_rm=1):
        """
        Perform per-sample random mixing by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (N, 4 , L, patch_size**2)
        """
        N, C, L, D = x.shape
       
        x = rearrange(x, 'n c l d -> (n l) c d')
        len_keep = 3 #int((C - channels_rm))
        
        noise = torch.rand(N*L, C, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N*L, C], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mixing_mask = rearrange(mask, '(n l) c -> n c l', n=N)

        x_masked = rearrange(x_masked, '(n l) c d -> n c l d', n=N)
        x_mix = self.unpatchify(x_masked)

        return x_mix, mixing_mask

    def random_location_masking(self, x, mixing_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x_masked: [N, C, L, D], sequence
        """
        N, L, D = x.shape
    
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask_l = torch.ones([N, L], device=x_masked.device)
        mask_l[:, :len_keep] = 0
        # unshuffle to get the binary mask_l
        mask_l = torch.gather(mask_l, dim=1, index=ids_restore)
        C = mixing_mask.shape[1]

        mask = mask_l.unsqueeze(-1).repeat(1, 1, C) + rearrange(mixing_mask, 'n c l -> n l c')
        mask[mask>1]=1.0

        mask = torch.einsum('nlc -> ncl', mask)
        return x_masked, mask_l, mask, ids_restore


    def forward_encoder(self, x, mix_channel): 
        
        # x = torch.cat([x1, x2, x3], 1)

        if mix_channel:
            x = self.patchify(x)
            x_mix, mixing_mask = self.random_channel_mixing(x)   # channel first to combine modality
            x = x_mix

        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # x_masked, mask_l, mask, ids_restore = self.random_location_masking(x, mixing_mask, mask_ratio)

        # x = x_masked
        
        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # return x, x_mix, mixing_mask, mask_l, mask, ids_restore
        return outcome

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        N, L, D = x.shape
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_decoder2(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed2(x)

        N, L, D = x.shape
        # append mask tokens to sequence
        mask_tokens = self.mask_token2.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed2

        # apply Transformer blocks
        for blk in self.decoder_blocks2:
            x = blk(x)
        x = self.decoder_norm2(x)

        # predictor projection
        x = self.decoder_pred2(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_decoder3(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed3(x)

        N, L, D = x.shape
        # append mask tokens to sequence
        mask_tokens = self.mask_token3.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed3

        # apply Transformer blocks
        for blk in self.decoder_blocks3:
            x = blk(x)
        x = self.decoder_norm3(x)

        # predictor projection
        x = self.decoder_pred3(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x1, x2, x3, pred1, pred2, pred3, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L, C], 0 is x1, 1 is x2, 
        """
        target = torch.cat([x1, x2, x3], 1)
        target = self.patchify(target) 
        N, C, L, D = target.shape
        
        pred = torch.cat([pred1, pred2, pred3], 2)
        preds = torch.split(pred, D, 2)
        pred = torch.stack(preds, 3)
        pred = torch.einsum('nldc->ncld', pred)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return pred, loss

#     def forward(self, imgs, depths, thermal, mask_ratio):
#         latent, x_mix, mixing_mask, mask_l, mask, ids_restore = self.forward_encoder(imgs, depths, thermal, mask_ratio)

#         # outcome = latent[:, 0] # class token ---
#         # self.head

#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         pred2 = self.forward_decoder2(latent, ids_restore)
#         pred3 = self.forward_decoder3(latent, ids_restore)

#         pred, loss = self.forward_loss(imgs, depths, thermal, pred, pred2, pred3, mask)
        
#         return loss, x_mix, mixing_mask, pred, mask_l, mask

    def forward(self, x, mix_channel=False):
        x = self.forward_encoder(x, mix_channel)
        x = self.head(x)
        return x

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = ChannelMaskedAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = ChannelMaskedAEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = ChannelMaskedAEViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks



def create_model(model_name, pth_path=None, in_chans=3):
    # create model
    method = eval(model_name)
    model = method(norm_pix_loss=False, in_chans=in_chans)

    if pth_path:
        logger = get_logger('model')
        checkpoint = torch.load(pth_path, map_location='cpu')

        logger.info("Load pre-trained checkpoint from: %s"%pth_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info("Loading Done!")

    return model
