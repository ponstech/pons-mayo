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

import torch
import torch.nn as nn

import timm.models.vision_transformer
from srcs.utils import get_logger
from srcs.utils.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from einops import rearrange
from .mul_ae2 import ChannelMaskedAEViT


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.head = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], kwargs['num_classes']),
        )
        # self.head = nn.Sequential(
        #     nn.Linear(kwargs['embed_dim'], 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, kwargs['num_classes']),
        # )
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
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
        x: (N, 4 , H, W)
        """
        x = self.patchify(x)
        N, C, L, D = x.shape
       
        x = rearrange(x, 'n c l d -> (n l) c d')
        len_keep = int((C - channels_rm))
        
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

    def forward_features(self, x, mix_channel):

        if mix_channel:
            x_mix, mixing_mask = self.random_channel_mixing(x)
            x = x_mix

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
        
    def forward(self, x, mix_channel=False):
        x = self.forward_features(x, mix_channel)
        x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16(**kwargs):
    model = ChannelMaskedAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small_patch16(**kwargs):  #cc
    model = ChannelMaskedAEViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16(**kwargs): #cc
    model = ChannelMaskedAEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def create_model(model_name, finetune, num_classes, drop_path):
    # create model
    method = eval(model_name)
    model = method(
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool=True,
    )
    # print(model)
    if finetune:
        logger = get_logger('model')
        checkpoint = torch.load(finetune, map_location='cpu')

        logger.info("Load pre-trained checkpoint from: %s" % finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']
        state_dict = model.state_dict()
        
        for k in ['head.weight', 'head.bias', 'head.0.weight', 'head.0.bias', 'head.3.weight', 'head.3.bias']:
            if k in checkpoint_model: # and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info("Loading Done!")

    
    # assert set(msg.missing_keys) == {'head.0.weight', 'head.0.bias', 'head.3.weight', 'head.3.bias', 'fc_norm.weight', 'fc_norm.bias'}
    trunc_normal_(model.head[0].weight, std=2e-5)
    # trunc_normal_(model.head[3].weight, std=2e-5)

    return model