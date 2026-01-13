import os
#Eğer deterministik mod tam oturmadıysa (örneğin env variable geç set edildiyse), sonuçlar 1–5 puan oynayabiliyor.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import numpy as np
import random
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
import hydra
from torchvision import transforms
from srcs.utils import instantiate, get_logger

from srcs.data_loader.data_loaders import AUDataLoader
from srcs.model.loss import bce_logit_loss
import torch.nn as nn

import srcs.utils.lrd as lrd
from torch.nn.parallel import DistributedDataParallel

project_dir = 'a'


def train_worker(config):

    """
    if config.seed:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
       # torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False #eskiden true idi...sonuclar farklılık gösterdi 
        
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        """

    logger = get_logger('train')
    # setup data_loader instances
    
    data_loader, valid_data_loader, test_loader = AUDataLoader(config).get_data_loaders()

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    model = model.to(config.local_rank)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # logger.info(trainable_params)
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')


    criterions = [instantiate(cri, config) for cri in config['criterions']]

    metrics = [instantiate(met, is_func=True) for met in config['metrics']]


    param_groups = lrd.param_groups_lrd(model, config.weight_decay,
                        no_weight_decay_list=model.no_weight_decay(),
                        layer_decay=config.layer_decay
                        )
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr)
    # logger.info(optimizer)

    #model = DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)

    loss_scaler = instantiate(config.loss_scaler)

    lr_adjust = instantiate(config.lr_adjust, config, optimizer=optimizer)
    trainer = instantiate(config.trainer, model, criterions, metrics, optimizer,
                            config,
                            data_loader,
                            valid_data_loader,
                            test_loader,
                            lr_adjust, loss_scaler)
    trainer.train()
#def __init__(self, model, criterions, metric_ftns, optimizer, config, data_loader,
                #  valid_data_loader=None, lr_scheduler=None, len_epoch=None):

"""
def init_worker(rank, ngpus, working_dir, project_dir, config, local_port):
    # initialize training config
    config = OmegaConf.create(config)
    config.local_rank = rank
    config.cwd = working_dir
    config.project_dir = project_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(local_port)
    dist.init_process_group(
        backend=config.backend,
        world_size=ngpus,
        rank=rank)

    train_worker(config)

 """
@hydra.main(config_path='conf/finetune/', config_name='us_finetune')
# @hydra.main(config_path='conf/', config_name='finetune')
def main(config):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids)
    #local_port = 10000 + random.randint(1000, 9000)
    
    

    n_gpu = torch.cuda.device_count()
    assert n_gpu, 'Can\'t find any GPU device on this machine.'

    # str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))
    
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)
    
    working_dir = str(Path.cwd())
    config = OmegaConf.create(config)
    config.local_rank = 0
    config.cwd = working_dir
    config.project_dir = project_dir
    
    train_worker(config)
    """
    torch.multiprocessing.spawn(init_worker, nprocs=n_gpu, args=(
        n_gpu, working_dir, project_dir, config, local_port))
        """


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    project_dir = str(Path.cwd())
    main()