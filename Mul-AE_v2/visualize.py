from ctypes import util
import logging
import torch
import hydra
from omegaconf import OmegaConf
from srcs.utils import instantiate
from srcs.data_loader.data_loaders import AUDataLoader
from srcs.utils.util import save_images
from srcs.logger import MetricLogger
from pathlib import Path

@hydra.main(config_path='conf', config_name='visualize')
def main(config):
    working_dir = str(Path.cwd())


    logger = logging.getLogger('visualize')    
    config = OmegaConf.to_yaml(config, resolve=True)
    config = OmegaConf.create(config)
    config.cwd = working_dir
    
    # setup data_loader instances
    data_loader = AUDataLoader(config).get_data_loaders()

    # restore network architecture and trained weights
    model = instantiate(config.arch)
    # logger.info(model)

    # instantiate metrics
    
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    metric_logger = MetricLogger(logger, delimiter="  ")

    with torch.no_grad():
        for i, out_dic in enumerate(metric_logger.log_every(data_loader, config.print_freq, f'Visual: ')):
            texture = out_dic['texture']
            depth = out_dic['depth']
            img_name = out_dic['name']
            ids_keep = out_dic['ids_keep']
            ids_restore =  out_dic['ids_restore']

            texture = texture.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            texture = texture.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            ids_keep = ids_keep.to(device, non_blocking=True)
            ids_restore = ids_restore.to(device, non_blocking=True)

            loss, x_mix, mask_mixing, pred, mask_l, mask = model(texture, depth, ids_keep, ids_restore, config.mask_ratio)

             # save img
            pred = model.unpatchify(pred) # (N, C, H, W)

            mask_mixing = mask_mixing.unsqueeze(-1).repeat(1, 1, 1, 16**2) # n l c d
            mask_mixing = model.unpatchify( mask_mixing)

            mask_l = mask_l.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 16**2) # n l c d
            mask_l = model.unpatchify(torch.einsum('nlcd->ncld', mask_l))
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, 16**2)    # n c l d    
            mask = model.unpatchify(mask)  # (N, C, H, W)

            # util.run_one_image(texture, depth, x_mix, pred, mask_l, mask, img_name, self.config.save_dir+f'/Epoch {epoch:03d}_{batch_idx:04d}.jpg')

            file_dirs = [config.save_dir+'/'+name for name in img_name]

            save_images(texture, depth, x_mix, mask_mixing, pred, mask_l, mask, file_dirs)

            
            

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
