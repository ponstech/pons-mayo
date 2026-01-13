from cProfile import label
import logging
import torch
import hydra
from omegaconf import OmegaConf
from srcs.utils import instantiate
from srcs.data_loader.data_loaders import AUDataLoader
from srcs.logger import BatchMetrics, MetricLogger, SmoothedValue
import numpy as np
import random
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd


def F1(true, pred):
    F1s = []
    precisions = []
    recalls = []
    if true.shape != pred.shape:
        print("Two array must have exactly the same dimension!!")
        return []
    for ix in range(true.shape[1]):
        F1s.append(f1_score(true[:, ix], pred[:, ix], zero_division=0))
        precisions.append(precision_score(true[:, ix], pred[:, ix],zero_division=0))
        recalls.append(recall_score(true[:, ix], pred[:, ix], zero_division=0))
    f1 = np.array(F1s, dtype=np.float32)
    precision = np.array(precisions, dtype=np.float32)
    recall = np.array(recalls, dtype=np.float32)
    return f1, np.mean(f1), precision, np.mean(precision), recall, np.mean(recall)

logger = logging.getLogger('evaluate')

@hydra.main(config_path='conf', config_name='evaluate')
def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids)
    if config.seed:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

    logger = logging.getLogger('evaluate')    

    # setup data_loader instances
    data_loader = AUDataLoader(config).get_data_loaders()

    device = 0
    # restore network architecture
     # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    model = model.to(device)
    # logger.info(model)


    # instantiate loss and metrics
    criterions = [instantiate(cri, config) for cri in config['criterions']]
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    criterions = [c.to(device) for c in criterions]


    
    metric_logger = MetricLogger(logger, delimiter="  ")
    model.eval()

    with torch.no_grad():
        all_preds, all_label = [], []
        for batch_idx, out_dic in enumerate(metric_logger.log_every(data_loader, 100, f'Test ')):
            texture, targets = out_dic['texture'], out_dic['label']
            targets = targets.to(device)
            texture = texture.to(device)
            
            
            # compute output
            output = model(texture, config.channel_mixing)
            loss = criterions[0](output, targets)
            
            preds = torch.where(output.sigmoid()>0.4, 1.0, 0.0)

            preds = preds.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            all_preds.append(preds)
            all_label.append(targets)

            # f1_arr, mean_f1, _, mean_prec, _, mean_recall = F1(preds, targets)

            metric_logger.update(loss=loss)
            # metric_logger.update(F1=torch.tensor(mean_f1))

            
        
        all_preds, all_label = np.concatenate(all_preds, 0), np.concatenate(all_label, 0)
        
        f1_arr, mean_f1, _, mean_prec, _, mean_recall = F1(all_preds, all_label)
        

        # labels = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU14','AU15','AU17','AU23','AU24']
        labels = ['AU1','AU2','AU4','AU6','AU9','AU12','AU25','AU26']
        dict_data = {}
        for j in range(len(f1_arr)):
            au_f1 = f1_arr[j]
            au_name = labels[j]
            dict_data[au_name] = [str(au_f1)]
        
        pd_data = pd.DataFrame(dict_data)
        pd_data.to_csv('./result.csv')
        
        
        logger.info(f'Test F1 ===> {mean_f1:.5f}')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
