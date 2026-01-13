# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import math

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


class Adjust_LearningRater():
    def __init__(self, config, lr_steps, lr_beta, is_cosine_decay, optimizer):
        self.epochs = config.epochs
        self.warmup_epochs = config.warmup_epochs
        self.lr_steps = lr_steps
        self.lr = config.lr
        self.min_lr = config.min_lr
        self.lr_beta = lr_beta
        self.optimizer = optimizer
        self.is_cosine_decay = is_cosine_decay
        self.decay_end = self.epochs 
        decay_epochs = config.decay_epochs if hasattr(config, 'decay_epochs') else config.warmup_epochs
        if isinstance(decay_epochs, int):
            self.decay_start = decay_epochs
        elif len(decay_epochs) == 1:
            self.decay_start = decay_epochs[0]
        else:
            self.decay_start, self.decay_end = decay_epochs
            

    def __call__(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch in self.lr_steps:
            self.lr *= self.lr_beta.pop(0)
        
        if self.warmup_epochs<0: # no warm up
            lr = self.lr
        elif epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            if self.is_cosine_decay:
                lr = self.lr
                if epoch>=self.decay_start and epoch<self.decay_end:
                    lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - self.decay_start) /
                    (self.decay_end - self.decay_start)))
                elif epoch>= self.decay_end:
                    lr = self.min_lr
            else:
                lr = self.lr
            
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr