#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
###################   SHOULD CHANGE EMBED DIM
###################   SHOULD CHANGE DATA NAMES AT DATA_LOADERS.PY 


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

import os
from pyexpat import model
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange
import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
#from src.transforms import make_transforms

#cc
import src.models.vision_transformer as vit
from timm.layers import trunc_normal_
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from timm.utils import accuracy

import torchvision.datasets as datasets
from torch.utils.data import RandomSampler, SequentialSampler
import src.utils.lr_decay as lrd
import src.utils.lr_sched as lr_sched
from src.data_loader.data_loaders import AUDataLoader
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --
nb_class = 2
_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed_all(_GLOBAL_SEED)
#torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def unpatchify(x):
        """
        x: (N, C , L, patch_size**2)
        img_depth: (N, C, H, W)
        """
        img_size=224
        patch_size=16
        in_chans=3
        embed_dim=384 #vit small
        #embed_dim = 1024  #vit large
        
        patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        N, C, L, D = x.shape
        p = patch_embed.patch_size[0]
        h = w = int(x.shape[2]**.5)
        assert h * w == x.shape[2]
        
        x = rearrange(x, 'n c (h w) (p q) -> n c (h p) (w q)', h=h, p=p)
        
        return x

def patchify(x):
    
        img_size=224
        patch_size=16
        in_chans=3
        embed_dim=384  #change me
        #embed_dim = 1024
        """
        imgs: (N, 4, H, W)
        x: (N, 4 , L, patch_size**2)
        """
        patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        N, C, H, W = x.shape
        p = patch_embed.patch_size[0]
        assert x.shape[2] == x.shape[3] and x.shape[2] % p == 0

        x = rearrange(x, 'n c (h p) (w q) -> n c (h w) (p q)', p=p, q=p)
        
        return x
    
def random_channel_mixing(x, channels_rm=1):
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
        x_mix = unpatchify(x_masked)

        return x_mix, mixing_mask
def top_module(x):
    x = patchify(x)
    x_mix, mixing_mask = random_channel_mixing(x) 
    
    return x_mix
    
def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
  
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    texture_root = args['data']['texture_root']
    depth_root = args['data']['depth_root']
    thermal_root = args['data']['thermal_root']
    csv_root = args['data']['csv_root']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = "/home/ec2-user/SageMaker/PRE_Checkpoints/channel_mix_ijepa/channel_mix-ijepaVITSMALL-ep30.pth.tar"
    #load_path = "/home/ec2-user/SageMaker/PRE_Checkpoints/channel_mix_ijepa/channel_mix_jepa_vit_large-ep30.pth.tar"
    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'Train loss'),
                           ('%.5f', 'Test loss'),
                           ('%.3f', 'Test - Acc@1'),
                           ('%.3f', 'Test - Acc@5'),
                           ('%d', 'Test time (ms)'),
                           ('%d', 'time (ms)'))
    
    #👽👽👽
    target_encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        nb_class=nb_class)
    
    
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = checkpoint['target_encoder']
    msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    trunc_normal_(target_encoder.head.weight, std=2e-5)
    #target_encoder.head = torch.nn.Sequential(torch.nn.BatchNorm1d(target_encoder.head.in_features, affine=False, eps=1e-6), target_encoder.head)
    target_encoder.to(device)    #
    #👽👽👽
    
 
    modalities = ['texture', 'depth', 'thermal']

    # -- init data-loaders/samplers
    supervised_loader_train, supervised_loader_val, supervised_loader_test = AUDataLoader(
                crop_size=crop_size,
                patch_size=patch_size,
                pred_mask_scale=pred_mask_scale,
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=aspect_ratio,
                num_enc_masks=num_enc_masks,
                num_pred_masks=num_pred_masks,
                allow_overlap=False,
                min_keep=min_keep,
                batch_size=batch_size,
                num_workers=num_workers,
                csv_root=csv_root,
                modalities=modalities,
                texture_root=texture_root,
                depth_root=depth_root,
                thermal_root=thermal_root
            ).get_data_loaders()

    
    
        
     #😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎😎
    ipe = len(supervised_loader_train)
    
    
    #🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶  
    """
     #original 
    param_groups = [
        {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        },  {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    
    """
    """
    
    # w/ constant wd 
    param_groups = [
        {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)),
            'weight_decay': wd
        },  {
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
   
    """
    """
    param_groups = [
    {'params': target_encoder.head.parameters(), 'lr': 1e-3, 'weight_decay': 0},
    {'params': target_encoder.blocks[-2:].parameters(), 'lr': 1e-4, 'weight_decay': 1e-5}
]
    """
    param_groups = lrd.param_groups_lrd(target_encoder, wd)
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    
    #optimizer = torch.optim.AdamW(param_groups)
    """
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*ipe),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*ipe))
    """
    """
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*ipe))
    """
   
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None  
    #🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶  
    criterion = torch.nn.CrossEntropyLoss() #cc
    print("criterion = %s" % str(criterion)) #cc
    
    
    """ #cc since target encoder will be finetuned without freezing, comment line
    for p in target_encoder.parameters():
        p.requires_grad = False
    """
    """
      #linear prob + global pool yöntemi 
    for p in target_encoder.parameters():
        p.requires_grad = False  

    # sonra sadece head'i aç
    for p in target_encoder.head.parameters():
        p.requires_grad = True
        
    # 3. Son 3 transformer bloğunu aç (opsiyonel)
    for p in target_encoder.blocks[-2:].parameters():
        p.requires_grad = True
   """
    # -- momentum schedule
    #momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
     #                     for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    accum_iter = 1 #cc
    # -- load training checkpoint

    
    #target_encoder = target_encoder.module #cc check if it includes module parameter
    
    def save_checkpoint(epoch):
        save_dict = {
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch}')) #cc  del epoch+1

    # -- TRAINING LOOP  
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        optimizer.zero_grad()
        # -- update distributed-data-loader epoch
        #unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        target_encoder.train(True)
        
        for itr, out_dict in enumerate(supervised_loader_train):

            def load_imgs():
                # -- unsupervised imgs
                #samples = sample.to(device, non_blocking=True)
                #targets = target.to(device, non_blocking=True)

                texture, targets, name = out_dict['texture'], out_dict['label'], out_dict['name']
                targets = targets.to(device, non_blocking=True)
                targets = targets.squeeze(1).long()
                depth = out_dict['depth']
                thermal = out_dict['thermal']
                texture = torch.cat([texture, depth, thermal], dim=1)
            
                texture = texture.to(device, non_blocking=True)
                x_mix = top_module(texture)
                
                return (x_mix, targets)
            
            imgs, targets = load_imgs()


            def train_step():
                if itr % accum_iter == 0:  
                    lr_sched.adjust_learning_rate(optimizer, itr / len(supervised_loader_train) + epoch, warmup, num_epochs, lr, final_lr)
                # _new_lr = scheduler.step()
                # _new_wd = wd_scheduler.step()
                # --

                def loss_fn(h, targets):
                    loss = criterion(h, targets)
                    #loss = AllReduce.apply(loss) #maede özel olarak demisski word size 1 den kucuksse gec, ama burada o yok. O yuzden ddp olmadıgında hata verıyor
                    # TODO: verify below.
                    # It is not necessary to use another allreduce to sum all loss. 
                    # Additional allreduce might have considerable negative impact on training speed.
                    # See: https://discuss.pytorch.org/t/distributeddataparallel-loss-compute-and-backpropogation/47205/4                    
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = target_encoder(imgs)
                    loss = loss_fn(h, targets)

                loss_value = loss.item()
                loss /= accum_iter
                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)  #iter accum icin asagı tasıdım
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(target_encoder.named_parameters()) 
                
                if (itr + 1) % accum_iter == 0:
                    optimizer.zero_grad()
                   

                return (float(loss))
            (loss), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))
                    """
                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

                    """
            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        testAcc1 = AverageMeter()
        testAcc2 = AverageMeter()
        test_loss = AverageMeter()

        # -- Save Checkpoint after every epoch
        from sklearn.metrics import confusion_matrix
        @torch.no_grad()
        def evaluate(unused_parameters=None):
            crossentropy = torch.nn.CrossEntropyLoss()
            num_classes = 2
            all_preds = []
            all_labels = []
            total_loss = 0.0
            total_samples = 0
            target_encoder.eval()              
            
            
            for cnt, out_dict in enumerate(supervised_loader_val):
                #images = samples.to(device, non_blocking=True)
                #labels = targets.to(device, non_blocking=True)

                texture, targets, name = out_dict['texture'], out_dict['label'], out_dict['name']
                targets = targets.to(device, non_blocking=True)
                targets = targets.squeeze(1).long()
                depth = out_dict['depth']
                thermal = out_dict['thermal']
                texture = torch.cat([texture, depth, thermal], dim=1)
            
                texture = texture.to(device, non_blocking=True)
                x_mix = top_module(texture)
                images = x_mix
                labels = targets
                                 
                with torch.cuda.amp.autocast():
                    output = target_encoder(images)
                    loss = crossentropy(output, labels)
                    
                preds = output.argmax(dim=1)
                probs = output.softmax(dim=1)   # <-- sınıf olasılıkları

                # tüm batch'i CPU float32'ye çevirip sakla
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
                total_loss += loss.item() * batch_size
                total_samples += batch_size
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
        
            # confusion matrix ve accuracy
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            acc = cm.diagonal().sum() / cm.sum() * 100
            avg_loss = total_loss / total_samples
        
            print(f"Test Accuracy: {acc:.2f}%")
            print(f"Test Loss: {avg_loss:.4f}")    # bu kısım hatali olabilir kontrol et.
            print("Confusion Matrix:")
            print(cm)  
        vtime = gpu_timer(evaluate)
        
        #logger.info('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Test loss {losses.avg:.3f}'.format(top1=testAcc1, top5=testAcc5, losses=test_loss))
        
        # -- Logging
        def log_test():
            csv_logger.log(epoch + 1, test_loss.val, testAcc1.avg, testAcc2.avg, vtime)
            if (itr % log_freq == 0) or np.isnan(test_loss) or np.isinf(test_loss): # TODO: fix TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

                logger.info('[%d] test_loss: %.3f '
                            ' - test_acc1 - [%.3f], test_acc2 - [%.3f]'
                            '[mem: %.2e] '
                            '(%.1f ms)'

                            % (epoch + 1,
                                test_loss.avg,
                                testAcc1.avg, testAcc2.avg,
                                torch.cuda.max_memory_allocated() / 1024.**2,
                                vtime))
        #log_test()

        # -- Save Checkpoint after every epoch
        logger.info('avg. train_loss %.3f' % loss_meter.avg)
        logger.info('avg. test_loss %.3f avg. Accuracy@1 %.3f - avg. Accuracy@2 %.3f' % (test_loss.avg, testAcc1.avg, testAcc2.avg))
        save_checkpoint(epoch+1)
        assert not np.isnan(loss), 'loss is nan'
        print('Loss:', loss)        

if __name__ == "__main__":
    main()