from posixpath import dirname
from PIL import Image
import yaml
import hydra
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from importlib import import_module
from itertools import repeat
from functools import partial, update_wrapper
import torch.nn as nn
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
       
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_logger(name=None):
    if is_master():
        # TODO: also configure logging for sub-processes(not master)
        hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in config, f'Config should have \'_target_\' for class instantiation.'
    target = config['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)

def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.01)

    return init_fun


def draw_image(ax, img, name):
    imagenet_mean = np.array([0.485, 0.456, 0.406], np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], np.float32)
    ax.imshow(torch.clip((img * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    ax.set_title(name, fontsize=12)

def draw_onechannel(ax, img, name, mean=IMAGENET_DEFAULT_MEAN[0], std=IMAGENET_DEFAULT_STD[0], camp='gray'):
    ax.imshow(torch.clip((img*std + mean) * 255, 0, 255).squeeze().int(), cmap=camp)
    ax.set_title(name, fontsize=12)

def get_img_from_torch(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    arr = torch.clip((img*std + mean) * 255, 0, 255).squeeze().int().numpy().astype(np.uint8)
    return Image.fromarray(arr)

def run_mae_image(x, y, mask, file_path):
    x = torch.einsum('nchw->nhwc', x).detach().cpu()
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    mask_img = x * (1 - mask)

    im_paste = x * (1 - mask) + y * mask

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(16, 8))


    # image is [H, W, 3]
    draw_image(ax1, x[0], 'texture')

    draw_image(ax2, mask_img[0], 'masked')

    draw_image(ax3, im_paste[0], 'reconstruction')

    fig.savefig(file_path)
    plt.cla()
    plt.close(fig)





def run_one_image(x1, x2, x3, x_mix, y, mask_l, mask, img_name, file_path):
    # (N, C, H, W)
    
    x1 = torch.einsum('nchw->nhwc', x1).detach().cpu()
    x2 = torch.einsum('nchw->nhwc', x2).detach().cpu()
    x3 = torch.einsum('nchw->nhwc', x3).detach().cpu()

    # run MAE
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    y1 = y[:, :, :, :3]
    y2 = y[:, :, :, 3:4]
    y3 = y[:, :, :, 4:]

    
    # visualize the mask
    # NLC 
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    mask_l = torch.einsum('nchw->nhwc', mask_l).detach().cpu()

    mask1 = mask[:, :, :, :3]
    mask2 = mask[:, :, :, 3:4]
    mask3 = mask[:, :, :, 4:]

    # masked image
    x_mix = torch.einsum('nchw->nhwc', x_mix).detach().cpu()
    # im_masked = x_mix * (1 - mask1)


    # MAE reconstruction pasted with visible patches
    im_paste = x1 * (1 - mask1) + y1 * mask1
    im_paste2 = x2 * (1 - mask2) + y2 * mask2
    im_paste3 = x3 * (1 - mask3) + y3 * mask3

    # make the plt figure larger
    fig, ((ax1, ax2, ax7, ax3), (ax4, ax5, ax6, ax8)) = plt.subplots(2, 4, figsize=(16, 8))

    fig.suptitle(img_name[0], fontsize=16)

    # image is [H, W, 3]
    draw_image(ax1, x1[0], 'texture')

    draw_onechannel(ax2, x2[0], 'depth')

    draw_onechannel(ax7, x3[0], 'thermal')
    
    draw_image(ax3, x_mix[0], 'mixed')

    draw_onechannel(ax4, 1-mask_l[0, :, :, 0], 'location mask') # mask: 0 is holding, 1 for removing

    # draw recon x1
    draw_image(ax5, im_paste[0], 'reconstruction texture')
    draw_onechannel(ax6, im_paste2[0], 'reconstruction depth')
    draw_onechannel(ax8, im_paste3[0], 'reconstruction thermal')
    
    fig.savefig(file_path)
    plt.cla()
    plt.close(fig)

     
def save_images(x1, x2, x_mix, mix_masking, y, mask_l, mask, file_dirs):

    # (N, C, H, W)
    N = x1.shape[0]
    x1 = torch.einsum('nchw->nhwc', x1).detach().cpu()
    x2 = torch.einsum('nchw->nhwc', x2).detach().cpu()

    # run MAE
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    y1 = y[:, :, :, :3]
    y2 = y[:, :, :, 3:]

    
    # visualize the mask
    # NLC 
    mix_masking = torch.einsum('nchw->nhwc', mix_masking).detach().cpu()

    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    mask_l = torch.einsum('nchw->nhwc', mask_l).detach().cpu()

    mask1 = mask[:, :, :, :3]
    mask2 = mask[:, :, :, 3:]

    # mixing image
    x_mix = torch.einsum('nchw->nhwc', x_mix).detach().cpu()

    # masked image
    x_masked = x_mix * (1 - mask_l)

    # MAE reconstruction pasted with visible patches
    im_paste = x1 * (1 - mask1) + y1 * mask1
    im_paste2 = x2 * (1 - mask2) + y2 * mask2


    ##### save to dir
    for idx in range(N):
        save_dir = file_dirs[idx]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # texture
        img = get_img_from_torch(x1[idx])
        img.save(os.path.join(save_dir,  '01_texture.jpg'))

        #depth
        img = get_img_from_torch(x2[idx], IMAGENET_DEFAULT_MEAN[0], IMAGENET_DEFAULT_STD[0])
        img.save(os.path.join(save_dir, '02_depth.jpg'))

        # mixing image
        img = get_img_from_torch(x_mix[idx])
        img.save(os.path.join(save_dir, '03_mixing.jpg'))

        # mixing mask
        img = get_img_from_torch(1-mix_masking[idx, :, :, 0], 0.0, 1.0)
        img.save(os.path.join(save_dir, '04_m_mask_r.jpg'))

        img = get_img_from_torch(1-mix_masking[idx, :, :, 1], 0.0, 1.0)
        img.save(os.path.join(save_dir, '05_m_mask_g.jpg'))

        img = get_img_from_torch(1-mix_masking[idx, :, :, 2], 0.0, 1.0)
        img.save(os.path.join(save_dir, '06_m_mask_b.jpg'))

        img = get_img_from_torch(1-mix_masking[idx, :, :, 3], 0.0, 1.0)
        img.save(os.path.join(save_dir, '07_m_mask_d.jpg'))

        # x_masked
        img = get_img_from_torch(x_masked[idx])
        img.save(os.path.join(save_dir, '08_x_masked.jpg'))

        # location mask
        img = img = get_img_from_torch(1-mask_l[idx, :, :, 0], 0.0, 1.0)
        img.save(os.path.join(save_dir, '09_l_mask.jpg'))

        # reconstruction
        img = get_img_from_torch(im_paste[idx])
        img.save(os.path.join(save_dir, '10_texture_rec.jpg'))

        img = get_img_from_torch(im_paste2[idx], IMAGENET_DEFAULT_MEAN[0], IMAGENET_DEFAULT_STD[0])
        img.save(os.path.join(save_dir, '11_depth_rec.jpg'))
