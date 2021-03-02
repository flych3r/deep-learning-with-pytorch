import argparse
import datetime
import hashlib
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import LunaDataset
from model import LunaModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import enumerate_with_estimate

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num-workers',
            help='Number of worker processes for background data loading',
            default=os.cpu_count(),
            type=int,
        )
        parser.add_argument(
            '--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument(
            '--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument(
            '--validation_cadence',
            help='Number of epochs to wait before starting validation',
            default=1,
            type=int,
        )
        parser.add_argument(
            '--augmented',
            help='Augment the training data.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--augment-flip',
            help='Randomly flips the data left-right, up-down, and front-back.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--augment-offset',
            help='Randomly offsets the data slightly along the X and Y axes.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--augment-scale',
            help='Randomly increases or decreases the size of the candidate.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--augment-rotate',
            help='Randomly rotates the data around the head-foot axis.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--augment-noise',
            help='Randomly adds noise to the data.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--balanced',
            help='Balance the training data to half positive, half negative.',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--finetune',
            help='Start finetuning from this model.',
            default='',
        )
        parser.add_argument(
            '--finetune-depth',
            help='Number of blocks (counted from the head) to include in finetuning',
            type=int,
            default=1,
        )
        parser.add_argument(
            '--tb-prefix',
            default='luna',
            help='Data prefix to use for Tensorboard run. Defaults to chapter.',
        )
        parser.add_argument(
            'comment',
            help='Comment suffix for Tensorboard run.',
            nargs='?',
            default='',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.trn_writer = None
        self.val_writer = None
        self.total_training_samples = 0
        self.validation_cadence = self.cli_args.validation_cadence

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))
        train_dl = self.init_data_loader(ratio=int(self.cli_args.balanced))
        val_dl = self.init_data_loader(is_val_set=True)

        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info('Epoch {} of {}, {}/{} batches of size {}*{}'.format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            train_metrics = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', train_metrics)
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                val_metrics = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', val_metrics)
                best_score = max(score, best_score)
                self.save_model('seg', epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def init_model(self):
        model = LunaModel()
        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(
                'Finetuning from {}, blocks {}'.format(
                    self.cli_args.finetune, ' '.join(finetune_blocks)
                )
            )
            model.load_state_dict(
                {
                    k: v for k, v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info('Using CUDA; {} devices.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_data_loader(self, is_val_set=False, ratio=0):
        train_ds = LunaDataset(
            val_stride=10,
            is_val_set=is_val_set,
            ratio=ratio
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dl

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls' + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls' + self.cli_args.comment
            )

    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )
        batch_iter = enumerate_with_estimate(
            train_dl,
            'E{} Training'.format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                train_metrics
            )
            loss_var.backward()
            self.optimizer.step()

        self.total_training_samples += len(train_dl.dataset)
        return train_metrics.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_with_estimate(
                val_dl,
                'E{} Validation '.format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics)

        return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def log_metrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classification_threshold=0.5,
    ):
        self.init_tensorboard_writers()
        log.info('E{} {}'.format(
            epoch_ndx,
            type(self).__name__,
        ))

        neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= classification_threshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= classification_threshold

        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        pos_count = int(pos_label_mask.sum())
        neg_count = int(neg_label_mask.sum())

        true_pos_count = int((pos_label_mask & pos_pred_mask).sum())
        true_neg_count = int((neg_label_mask & neg_pred_mask).sum())

        false_pos_count = pos_count - true_pos_count
        false_neg_count = neg_count - true_neg_count

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean()

        metrics_dict['correct/all'] = (
            (true_pos_count + true_neg_count) / np.float32(metrics_t.shape[1]) * 100
        )
        metrics_dict['correct/neg'] = true_neg_count / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = true_pos_count / np.float32(pos_count) * 100

        precision = metrics_dict['pr/precision'] = \
            true_pos_count / np.float32(true_pos_count + false_pos_count)
        recall = metrics_dict['pr/recall'] = \
            true_pos_count / np.float32(true_pos_count + false_neg_count)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0)
        tpr = (metrics_t[None, METRICS_PRED_NDX, pos_label_mask] >= threshold[:, None])
        tpr = tpr.sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_NDX, neg_label_mask] >= threshold[:, None])
        fpr = fpr.sum(1).float() / neg_count

        log.info(
            'E{} {:8} {loss/all:.4f} loss, '
            '{correct/all:-5.1f}% correct, '
            '{pr/precision:.4f} precision, '
            '{pr/recall:.4f} recall, '
            '{pr/f1_score:.4f} f1 score'
            .format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples)

        fig = plt.figure()
        plt.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.total_training_samples)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.total_training_samples,
        )

        bins = [x / 50.0 for x in range(51)]

        neg_hist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if neg_hist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, neg_hist_mask],
                self.total_training_samples,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, pos_hist_mask],
                self.total_training_samples,
                bins=bins,
            )

        score = metrics_dict['pr/recall']
        return score

    def save_model(self, type_str, epoch_ndx, is_best=False):
        file_path = os.path.join(
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.total_training_samples,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'total_training_samples': self.total_training_samples,
        }
        torch.save(state, file_path)

        log.info('Saved model params to {}'.format(file_path))

        if is_best:
            best_path = os.path.join(
                'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state'
            )
            shutil.copyfile(file_path, best_path)

            log.info('Saved model params to {}'.format(best_path))

        with open(file_path, 'rb') as f:
            log.info('SHA1: ' + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    LunaTrainingApp().main()
