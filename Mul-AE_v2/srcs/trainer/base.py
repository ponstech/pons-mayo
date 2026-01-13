import os
import signal
import torch
import torch.distributed as dist
from abc import abstractmethod, ABCMeta
from pathlib import Path
import shutil
from numpy import inf

from srcs.utils import write_conf, is_master, get_logger
from srcs.logger import TensorboardWriter, EpochMetrics




class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config):
        self.config = config
        self.logger = get_logger('trainer')

        self.device = config.local_rank
        self.model = model
        self.optimizer = optimizer

        self.metric_ftns = metric_ftns

        
        self.epochs = config['epochs']

        # setup metric monitoring for monitoring model performance and saving best-checkpoint
        self.monitor = config.get('monitor', 'off')

        metric_names = ['loss'] + [met.__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(metric_names, phases=('train', 'valid'), monitoring=self.monitor)

        self.checkpt_top_k = config.get('save_topk', -1)
        self.early_stop = config.get('early_stop', inf)

        write_conf(self.config, 'config.yaml')

        self.start_epoch = 0
        self.checkpt_dir = Path(self.config.save_dir)
        log_dir = Path(self.config.log_dir)
        if is_master():
            if not os.path.exists(self.checkpt_dir): self.checkpt_dir.mkdir()
            # setup visualization writer instance
            if not os.path.exists(log_dir): log_dir.mkdir()
            self.writer = TensorboardWriter(log_dir, config['tensorboard'])
            # shutil.copytree(config.project_dir, os.path.basename(config.project_dir))
        else:
            self.writer = TensorboardWriter(log_dir, False)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        acc_valid = 0
        for epoch in range(self.start_epoch, self.epochs):
            result = self._train_epoch(epoch)
            
            if self.device == 0:
                self.ep_metrics.update(epoch, result)

                # print result metrics of this epoch
                # max_line_width = max(len(line) for line in str(self.ep_metrics).splitlines()) 
                # divider ---
                # self.logger.info('-'*max_line_width)
                # self.logger.info(str(self.ep_metrics.latest()) + '\n')

                # check if model performance improved or not, for early stopping and topk saving
                if float(result['accuracy/valid']) > acc_valid:
                    acc_valid = float(result['accuracy/valid'])
                    self._save_checkpoint(epoch+1, save_best=True)
                    print(f'Saved best model at epoch: {epoch}')
                is_best = False
                improved = self.ep_metrics.is_improved()
                if improved:
                    not_improved_count = 0
                    # is_best = True
                    # print('best_epoch:', epoch)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop and is_master():
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                    os.kill(os.getppid(), signal.SIGTERM)

                using_topk_save = self.checkpt_top_k > 0
                if (epoch+1) % self.config.save_epochs == 0:
                    self._save_checkpoint(epoch+1, save_best=is_best, save_latest=using_topk_save)
                # keep top-k checkpoints only, using monitoring metrics
                if using_topk_save:
                    self.ep_metrics.keep_topk_checkpt(self.checkpt_dir, self.checkpt_top_k)

                self.ep_metrics.to_csv('epoch-results.csv')

                # divider ===
                self.logger.info('='*60)
            #dist.barrier()


    def _save_checkpoint(self, epoch, save_best=False, save_latest=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        """
        state = {
                'state_dict': self.model.state_dict(),
            }
        if save_best:
            filename = str(self.checkpt_dir / f'model_best.pth')
        else:
            filename = str(self.checkpt_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Model checkpoint saved at: \n    {self.config.cwd}/{filename}")
        

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = self.config.resume
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # TODO: support overriding monitor-metric config
        self.ep_metrics = checkpoint['epoch_metrics']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['_target_'] != self.config['optimizer']['_target_']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")