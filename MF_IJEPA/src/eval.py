#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
############################################    SHOULD CHANGE EMBED DIM!
############################################    SHOULD CHANGE lOAD PATH!

import os
import csv
from pyexpat import model
from sklearn.metrics import confusion_matrix
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
from src.data_loader.data_loaders import AUDataLoader
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
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange
import src.utils.lr_sched as lr_sched
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --
nb_class = 2
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
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
        embed_dim= 384
        
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
        embed_dim= 384  #change me
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
        
    load_path = "/home/ec2-user/SageMaker/channel_mix_i-jepa/log/jepa-ep9.pth.tar" #change me 
    
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
    
    #target_encoder.head = torch.nn.Sequential(torch.nn.BatchNorm1d(target_encoder.head.in_features,   affine=False, eps=1e-6), target_encoder.head)
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = checkpoint['target_encoder']
    msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    #trunc_normal_(target_encoder.head.weight, std=2e-5)
    target_encoder.to(device)    #bundan önce checkpointi yüklememiz lazım
  
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

    ipe = len(supervised_loader_train)
    
    
    #🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶🎶  
    """ original 
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
    # w/ constant wd 
    """
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
    param_groups = lrd.param_groups_lrd(target_encoder, wd)
   
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    logger.info('Using AdamW')
    
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
   
    testAcc1 = AverageMeter()
    testAcc2 = AverageMeter()
    test_loss = AverageMeter()
    

    
        # -- Save Checkpoint after every epoch
    @torch.no_grad()
    def evaluate(unused_parameters=None):
            crossentropy = torch.nn.CrossEntropyLoss()
            num_classes = 2
            all_preds = []
            all_labels = []
            all_csv_rows = [] 
            total_loss = 0.0
            total_samples = 0
            num_classes = 2
            target_encoder.eval()              
            
            for cnt, out_dict in enumerate(supervised_loader_test):
                #images = samples.to(device, non_blocking=True)
                #labels = targets.to(device, non_blocking=True)
                #batch_size = images.size(0)

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

                preds_ = preds.cpu()
                labels_ = labels.cpu()
                probs_ = probs.cpu()
                names_ = list(name)  # string list

                 # batch’i CSV listesine ekle
                for i in range(len(names_)):
                    all_csv_rows.append([
                        names_[i],
                        int(labels_[i].item()),        # True Diagnosis
                        int(preds_[i].item()),         # Final Diagnosis
                        float(probs_[i][0].item()),    # prob class 0
                        float(probs_[i][1].item())     # prob class 1
                    ])
                

                # tüm batch'i CPU float32'ye çevirip sakla
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
                #total_loss += loss.item() * batch_size
                #total_samples += batch_size
                
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
        
            # confusion matrix ve accuracy
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            acc = cm.diagonal().sum() / cm.sum() * 100
            #avg_loss = total_loss / total_samples
        
            print(f"Test Accuracy: {acc:.2f}%")
            #print(f"Test Loss: {avg_loss:.4f}")
            print("Confusion Matrix:")
            print(cm)    
            header_csv = ["img name", "True Diagnosis", "Final Diagnosis",
                  "pred_prob_class_0", "pred_prob_class_1"]

            save_path = "/home/ec2-user/SageMaker/channel_mix_i-jepa/ijepa_channel-mix_ft/ijepa/src/test_predictions.csv"   

            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header_csv)
                writer.writerows(all_csv_rows)
        
    vtime = gpu_timer(evaluate)
        
        #logger.info('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Test loss {losses.avg:.3f}'.format(top1=testAcc1, top5=testAcc5, losses=test_loss))
        
        # -- Logging  zaten asagidakini calistirmiyoruz log testi cagırmadık hiçbir yerde
       
        # -- Save Checkpoint after every epoch
        
    

    exit(0)
    
    """ #cc since target encoder will be finetuned without freezing, comment line
    for p in target_encoder.parameters():
        p.requires_grad = False
    """
    
    """  #linear prob + global pool yöntemi 
    for p in model.parameters():
    p.requires_grad = False  

    # sonra sadece head'i aç
    for p in model.head.parameters():
        p.requires_grad = True
    """
    
if __name__ == "__main__":
    main()