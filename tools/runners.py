import copy
import datetime
import itertools
import os
import gc
import random
import re
import time
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from tools.datasets import (TSEHeadTailDataset, TSEHeadTailDatasetV2,
                            TSEHeadTailDatasetV3, TSEHeadTailDatasetV4,
                            TSEHeadTailSegmentationDataset,
                            TSEHeadTailSegmentationDatasetV3,
                            TSEHeadTailSegmentationDatasetV4,
                            TSESegmentationDataset)
from tools.loggers import myLogger
from tools.losses import dist_loss, lovasz_hinge
from tools.metrics import jaccard
from tools.models import (
    EMA, BertModelWBinaryMultiLabelClassifierHead,
    BertModelWDualMultiClassClassifierAndSegmentationHead,
    BertModelWDualMultiClassClassifierHead, RobertaModelHeadClassAndAnchorHead,
    RobertaModelWDualMultiClassClassifierAndCumsumSegmentationHead,
    RobertaModelWDualMultiClassClassifierAndCumsumSegmentationHeadV2,
    RobertaModelWDualMultiClassClassifierAndSegmentationHead,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV4,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV5,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV6,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV7,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV8,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV9,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV10,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV11,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV12,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV13,
    RobertaModelWDualMultiClassClassifierAndSegmentationHeadV14,
    RobertaModelWDualMultiClassClassifierHead,
    RobertaModelWDualMultiClassClassifierHeadV2,
    RobertaModelWDualMultiClassClassifierHeadV3,
    RobertaModelWDualMultiClassClassifierHeadV4,
    RobertaModelWDualMultiClassClassifierHeadV5,
    RobertaModelWDualMultiClassClassifierHeadV6, SoftArgmax1D)
from tools.schedulers import pass_scheduler
from tools.splitters import mySplitter
from torch.nn import (BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss,
                      Sigmoid, Softmax)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (RandomSampler, SequentialSampler,
                                      WeightedRandomSampler)
from transformers import RobertaForMaskedLM

random.seed(71)
torch.manual_seed(71)


class Runner(object):
    def __init__(self, exp_id, checkpoint, device,
                 debug, config, default_config):
        # set logger
        self.exp_time = datetime\
            .datetime.now()\
            .strftime('%Y-%m-%d-%H-%M-%S')

        self.exp_id = exp_id
        self.checkpoint = checkpoint
        self.device = device
        self.debug = debug
        # self.logger = myLogger(f'./logs/{self.exp_id}_{self.exp_time}.log')
        self.logger = myLogger(f'./logs/{self.exp_id}.log')

        # set default configs
        self._fill_config_by_default_config(config, default_config)

        self.logger.info(f'exp_id: {exp_id}')
        self.logger.info(f'checkpoint: {checkpoint}')
        self.logger.info(f'debug: {debug}')
        self.logger.info(f'config: {config}')

        # unpack config info
        if 'description' in config:
            self.description = config['description']
        else:
            self.description = 'no description'
        # uppercase means raaaw value
        self.cfg_SINGLE_FOLD = config['SINGLE_FOLD']
        # self.cfg_batch_size = config['batch_size']
        # self.cfg_max_epoch = config['max_epoch']
        self.cfg_split = config['split']
        self.cfg_loader = config['loader']
        self.cfg_dataset = config['dataset']
        self.cfg_fobj = config['fobj']
        self.cfg_fobj_index_diff = config['fobj_index_diff']
        self.cfg_fobj_segmentation = config['fobj_segmentation']
        self.cfg_model = config['model']
        self.cfg_optimizer = config['optimizer']
        self.cfg_scheduler = config['scheduler']
        self.cfg_train = config['train']
        self.cfg_predict = config['predict']
        self.cfg_invalid_labels = config['invalid_labels'] \
            if 'invalid_labels' in config else None

        self.histories = {
            'train_loss': [],
            'valid_loss': [],
            'valid_acc': [],
        }

    def _fill_config_by_default_config(self, config, default_config):
        for (d_key, d_value) in default_config.items():
            if d_key not in config:
                message = f' --- fill {d_key} by dafault values, {d_value} ! --- '
                self.logger.warning(message)
                config[d_key] = d_value
            elif isinstance(d_value, dict):
                self._fill_config_by_default_config(config[d_key], d_value)

    def MLM(self):
        # load and preprocess train.csv
        trn_df = pd.read_csv(
            './inputs/datasets/w_private/tweet_sentiment_origin_merged_guchio.csv')
        trn_df = trn_df[trn_df.text.notnull()].reset_index(drop=True)

        if self.debug:
            trn_df = trn_df.sample(self.cfg_loader['trn_batch_size'] * 3,
                                   random_state=71)

        # build loader
        fold_trn_df = trn_df
        trn_loader = self._build_loader(mode='train', df=fold_trn_df,
                                        **self.cfg_loader)

        model = torch.nn.DataParallel(
            RobertaForMaskedLM.from_pretrained('roberta-base'))
        module = model.module
        resized_res = module.resize_token_embeddings(
            len(trn_loader.dataset.tokenizer))  # for sentiment
        self.logger.info(f'resized_res: {resized_res}')
        optimizer = self._get_optimizer(model=model, **self.cfg_optimizer)
        scheduler = self._get_scheduler(optimizer=optimizer,
                                        max_epoch=self.cfg_train['max_epoch'],
                                        **self.cfg_scheduler)
        iter_epochs = range(0, self.cfg_train['max_epoch'], 1)

        self.logger.info('start trainging !')
        for current_epoch in iter_epochs:
            model = model.to(self.device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            if isinstance(self.cfg_train['accum_mod'], int):
                accum_mod = self.cfg_train['accum_mod']
            elif isinstance(self.cfg_train['accum_mod'], list):
                accum_mod = self.cfg_train['accum_mod'][current_epoch]
            else:
                raise NotImplementedError('accum_mod')

            running_loss = 0.
            for batch_i, batch in enumerate(tqdm(trn_loader)):
                mlm_input_ids = batch['mlm_input_ids']
                mlm_labels = batch['mlm_labels']
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = model(mlm_input_ids,
                                masked_lm_labels=mlm_labels,
                                attention_mask=attention_mask)
                loss, prediction_scores = outputs[:2]
                loss.backward()
                running_loss += loss.item()

                if (batch_i + 1) % accum_mod == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            trn_loss = running_loss / len(trn_loader)

            self.logger.info(
                f'epoch: {current_epoch} / '
                + f'trn loss: {trn_loss:.5f} / '
                + f'lr: {optimizer.param_groups[0]["lr"]:.6f} / '
                + f'accum_mod: {accum_mod}')

            model = model.to('cpu')
            scheduler.step()

        if not os.path.exists(f'./inputs/datasets/pretrain/{self.exp_id}'):
            os.mkdir(f'./inputs/datasets/pretrain/{self.exp_id}')
        model.module.save_pretrained(
            f'./inputs/datasets/pretrain/{self.exp_id}/')

    def train(self):
        trn_start_time = time.time()
        # load and preprocess train.csv
        trn_df = pd.read_csv('./inputs/origin/train.csv')
        trn_df = trn_df[trn_df.text.notnull()].reset_index(drop=True)

        # split data
        splitter = mySplitter(**self.cfg_split, logger=self.logger)
        fold = splitter.split(
            trn_df['textID'],
            trn_df['sentiment'],
            group=trn_df['sentiment']
        )

        # load and apply checkpoint if needed
        if self.checkpoint:
            self.logger.info(f'loading checkpoint from {self.checkpoint} ...')
            checkpoint = torch.load(self.checkpoint)
            checkpoint_fold_num = checkpoint['fold_num']
            self.histories = checkpoint['histories']

        for fold_num, (trn_idx, val_idx) in enumerate(fold):
            if (self.checkpoint and fold_num < checkpoint_fold_num) \
               or (self.checkpoint and fold_num == checkpoint_fold_num
                   and checkpoint_fold_num == self.cfg_train['max_epoch'] - 1):
                self.logger.info(f'pass fold {fold_num}')
                continue

            if fold_num not in self.histories:
                self.histories[fold_num] = {
                    'trn_loss': [],
                    'val_loss': [],
                    'val_jac': [],
                }

            if self.debug:
                trn_idx = trn_idx[:self.cfg_loader['trn_batch_size'] * 3]
                val_idx = val_idx[:self.cfg_loader['tst_batch_size'] * 3]

            # build loader
            fold_trn_df = trn_df.iloc[trn_idx]
            if self.cfg_invalid_labels:
                fold_trn_textIDs = fold_trn_df.textID.tolist()
                fold_trn_df = fold_trn_df.set_index('textID')
                for invalid_label_csv in self.cfg_invalid_labels:
                    invalid_label_df = pd.read_csv(invalid_label_csv)
                    invalid_label_df = invalid_label_df\
                        .query(f'textID in {fold_trn_textIDs}')
                    for i, row in invalid_label_df.iterrows():
                        fold_trn_df.loc[row['textID'], 'selected_text'] = \
                            row['guchio_selected_text']
                fold_trn_df = fold_trn_df.reset_index()

            if 'rm_neutral' in self.cfg_train \
                    and self.cfg_train['rm_neutral']:
                fold_trn_df = fold_trn_df.query('sentiment != "neutral"')

            if self.cfg_train['pseudo']:
                fold_trn_df = pd.concat([fold_trn_df,
                                         pd.read_csv(self.cfg_train['pseudo'][fold_num])],
                                        axis=0).reset_index(drop=True)

            invalid_text_ids = [
                '4c279acff6',
                '96ff964db0',
                'eaf2942ee8',
                '12f21c8f19',
                '09d0f8f088',
                '3a906c871f',
                '780c673bca',
            ]
            fold_trn_df = fold_trn_df.query(
                f'textID not in {invalid_text_ids}')

            trn_loader = self._build_loader(mode='train', df=fold_trn_df,
                                            **self.cfg_loader)
            fold_val_df = trn_df.iloc[val_idx]
            val_loader = self._build_loader(mode='test', df=fold_val_df,
                                            **self.cfg_loader)

            # get fobj
            fobj = self._get_fobj(**self.cfg_fobj)
            fobj_segmentation = self._get_fobj(**self.cfg_fobj_segmentation)
            fobj_index_diff = self._get_fobj(**self.cfg_fobj_index_diff)
            if 'segmentation_loss_ratios' in self.cfg_train:
                if isinstance(self.cfg_train['segmentation_loss_ratios'], int):
                    segmentation_loss_ratios = [self.cfg_train['segmentation_loss_ratios']] \
                        * self.cfg_train['max_epoch']
                elif isinstance(self.cfg_train['segmentation_loss_ratios'], float):
                    segmentation_loss_ratios = [self.cfg_train['segmentation_loss_ratios']] \
                        * self.cfg_train['max_epoch']
                elif isinstance(self.cfg_train['segmentation_loss_ratios'], list):
                    segmentation_loss_ratios = self.cfg_train['segmentation_loss_ratios']
                else:
                    raise NotImplementedError('segmentation_loss_ratios')
            else:
                self.logger.warning('use default segmentation_loss_ratios')
                segmentation_loss_ratios = [1] * self.cfg_train['max_epoch']

            # build model and related objects
            # these objects have state
            model = self._get_model(**self.cfg_model)
            module = model if self.device == 'cpu' else model.module
            module.resize_token_embeddings(
                len(trn_loader.dataset.tokenizer))  # for sentiment
            optimizer = self._get_optimizer(model=model, **self.cfg_optimizer)
            scheduler = self._get_scheduler(optimizer=optimizer,
                                            max_epoch=self.cfg_train['max_epoch'],
                                            **self.cfg_scheduler)
            if self.checkpoint and checkpoint_fold_num == fold_num:
                module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                checkpoint_epoch = checkpoint['current_epoch']
                iter_epochs = range(checkpoint_epoch,
                                    self.cfg_train['max_epoch'], 1)
            else:
                checkpoint_epoch = -1
                iter_epochs = range(0, self.cfg_train['max_epoch'], 1)

            epoch_start_time = time.time()
            epoch_best_jaccard = -1
            self.logger.info('start trainging !')
            for current_epoch in iter_epochs:
                if self.checkpoint and current_epoch <= checkpoint_epoch:
                    print(f'pass epoch {current_epoch}')
                    continue

                start_time = time.time()
                # send to device
                model = model.to(self.device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                self._warmup(current_epoch, self.cfg_train['warmup_epoch'],
                             model)

                warmup_batch = self.cfg_train['warmup_batch'] if current_epoch == 0 else 0

                ema_model = copy.deepcopy(model)
                ema_model.eval()
                ema = EMA(model=ema_model,
                          mu=self.cfg_train['ema_mu'],
                          level=self.cfg_train['ema_level'],
                          n=self.cfg_train['ema_n'])

                if isinstance(self.cfg_train['accum_mod'], int):
                    accum_mod = self.cfg_train['accum_mod']
                elif isinstance(self.cfg_train['accum_mod'], list):
                    accum_mod = self.cfg_train['accum_mod'][current_epoch]
                else:
                    raise NotImplementedError('accum_mod')

                use_special_mask = self.cfg_train['use_special_mask']
                segmentation_loss_ratio = segmentation_loss_ratios[current_epoch]

                trn_loss = self._train_loop(
                    model, optimizer, fobj, trn_loader, warmup_batch,
                    ema, accum_mod, use_special_mask,
                    fobj_segmentation, segmentation_loss_ratio,
                    self.cfg_train['loss_weight_type'], fobj_index_diff,
                    self.cfg_train['use_dist_loss'],
                    self.cfg_train['single_word'])
                ema.on_epoch_end(model)
                if self.cfg_train['ema_n'] > 0:
                    ema.set_weights(ema_model)  # NOTE: model?
                else:
                    ema_model = model
                use_offsets = self.cfg_predict['use_offsets']
                val_loss, best_thresh, best_jaccard, val_textIDs, \
                    val_input_ids, val_preds, val_labels = \
                    self._valid_loop(ema_model, fobj, val_loader,
                                     use_special_mask, use_offsets,
                                     self.cfg_train['loss_weight_type'],
                                     self.cfg_predict['single_word'])
                epoch_best_jaccard = max(epoch_best_jaccard, best_jaccard)

                self.logger.info(
                    f'epoch: {current_epoch} / '
                    + f'trn loss: {trn_loss:.5f} / '
                    + f'val loss: {val_loss:.5f} / '
                    + f'best val thresh: {best_thresh:.5f} / '
                    + f'best val jaccard: {best_jaccard:.5f} / '
                    + f'lr: {optimizer.param_groups[0]["lr"]:.6f} / '
                    + f'accum_mod: {accum_mod} / '
                    + f'time: {int(time.time()-start_time)}sec')

                self.histories[fold_num]['trn_loss'].append(trn_loss)
                self.histories[fold_num]['val_loss'].append(val_loss)
                self.histories[fold_num]['val_jac'].append(best_jaccard)

                scheduler.step()

                # send to cpu
                ema_model = ema_model.to('cpu')
                # model = model.to('cpu')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

                self._save_checkpoint(fold_num, current_epoch,
                                      ema_model, optimizer, scheduler,
                                      # model, optimizer, scheduler,
                                      val_textIDs, val_input_ids, val_preds,
                                      val_labels, val_loss,
                                      best_thresh, best_jaccard)

            best_filename = self._search_best_filename(fold_num)
            if not os.path.exists(f'./checkpoints/{self.exp_id}/best'):
                os.mkdir(f'./checkpoints/{self.exp_id}/best')
            os.rename(
                best_filename,
                f'./checkpoints/{self.exp_id}/best/{best_filename.split("/")[-1]}')
            left_files = glob(f'./checkpoints/{self.exp_id}/{fold_num}/*')
            for left_file in left_files:
                os.remove(left_file)

            fold_time = int(time.time() - epoch_start_time) // 60
            line_message = f'{self.exp_id}: {self.description} \n' \
                f'fini fold {fold_num} in {fold_time} min. \n' \
                f'epoch best jaccard: {epoch_best_jaccard}'
            self.logger.send_line_notification(line_message)

            if self.cfg_SINGLE_FOLD:
                break

        fold_best_jacs = []
        for fold_num in range(self.cfg_split['split_num']):
            fold_best_jacs.append(max(self.histories[fold_num]['val_jac']))
        jac_mean = np.mean(fold_best_jacs)
        jac_std = np.std(fold_best_jacs)

        trn_time = int(time.time() - trn_start_time) // 60
        line_message = \
            f'----------------------- \n' \
            f'{self.exp_id}: {self.description} \n' \
            f'jaccard      : {jac_mean:.5f}+-{jac_std:.5f} \n' \
            f'best_jacs    : {fold_best_jacs} \n' \
            f'time         : {trn_time} min \n' \
            f'-----------------------'
        self.logger.send_line_notification(line_message)

    def _get_fobj(self, fobj_type):
        if fobj_type == 'bce':
            fobj = BCEWithLogitsLoss()
        elif fobj_type == 'bce_raw':
            fobj = BCELoss(reduction='mean')
        elif fobj_type == 'ce':
            fobj = CrossEntropyLoss()
        elif fobj_type == 'ce_noreduction':
            fobj = CrossEntropyLoss(reduction='none')
        elif fobj_type == 'lovasz':
            fobj = lovasz_hinge
        elif fobj_type == 'mse':
            fobj = MSELoss()
        elif fobj_type == 'nothing':
            fobj = None
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, num_output_units,
                   pretrained_model_name_or_path):
        if model_type == 'bert-segmentation':
            model = BertModelWBinaryMultiLabelClassifierHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'bert-headtail':
            model = BertModelWDualMultiClassClassifierHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail':
            model = RobertaModelWDualMultiClassClassifierHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-v2':
            model = RobertaModelWDualMultiClassClassifierHeadV2(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-v3':
            model = RobertaModelWDualMultiClassClassifierHeadV3(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-v4':
            model = RobertaModelWDualMultiClassClassifierHeadV4(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-v5':
            model = RobertaModelWDualMultiClassClassifierHeadV5(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-v6':
            model = RobertaModelWDualMultiClassClassifierHeadV6(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-head-anchor':
            model = RobertaModelHeadClassAndAnchorHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'bert-headtail-segmentation':
            model = BertModelWDualMultiClassClassifierAndSegmentationHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-cumsum-segmentation':
            model = RobertaModelWDualMultiClassClassifierAndCumsumSegmentationHead(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-cumsum-segmentation-v2':
            model = RobertaModelWDualMultiClassClassifierAndCumsumSegmentationHeadV2(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v4':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV4(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v5':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV5(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v6':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV6(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v7':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV7(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v8':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV8(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v9':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV9(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v10':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV10(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v11':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV11(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v12':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV12(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v13':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV13(
                num_output_units,
                pretrained_model_name_or_path
            )
        elif model_type == 'roberta-headtail-segmentation-v14':
            model = RobertaModelWDualMultiClassClassifierAndSegmentationHeadV14(
                num_output_units,
                pretrained_model_name_or_path
            )
        else:
            raise Exception(f'invalid model_type: {model_type}')
        if self.device == 'cpu':
            return model
        else:
            return torch.nn.DataParallel(model)

    def _get_optimizer(self, optim_type, lr, model):
        if optim_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                # weight_decay=1e-4,
                nesterov=True,
            )
        elif optim_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
            )
        elif optim_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=lr,
                momentum=0.9,
            )
        else:
            raise Exception(f'invalid optim_type: {optim_type}')
        return optimizer

    def _get_scheduler(self, scheduler_type, max_epoch,
                       optimizer, every_step_unit, cosine_eta_min,
                       multistep_milestones, multistep_gamma):
        if scheduler_type == 'pass':
            scheduler = pass_scheduler()
        elif scheduler_type == 'every_step':
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: every_step_unit**epoch,
            )
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=multistep_milestones,
                gamma=multistep_gamma
            )
        elif scheduler_type == 'cosine':
            # scheduler examples:
            #     [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
            # if you want to use cosine annealing, use below scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epoch - 1, eta_min=cosine_eta_min
            )
        else:
            raise Exception(f'invalid scheduler_type: {scheduler_type}')
        return scheduler

    def _build_loader(self, mode, df,
                      trn_sampler_type, trn_batch_size,
                      tst_sampler_type, tst_batch_size,
                      dataset_type, neutral_weight=1.,
                      longer_posneg_rate=1.):
        if mode == 'train':
            sampler_type = trn_sampler_type
            batch_size = trn_batch_size
            drop_last = True
        elif mode == 'test':
            sampler_type = tst_sampler_type
            batch_size = tst_batch_size
            drop_last = False
        else:
            raise NotImplementedError('mode {mode} is not valid for loader')

        if dataset_type == 'tse_segmentation_dataset':
            dataset = TSESegmentationDataset(mode=mode, df=df,
                                             logger=self.logger,
                                             debug=self.debug,
                                             **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_dataset':
            dataset = TSEHeadTailDataset(mode=mode, df=df, logger=self.logger,
                                         debug=self.debug, **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_dataset_v2':
            dataset = TSEHeadTailDatasetV2(mode=mode, df=df,
                                           logger=self.logger,
                                           debug=self.debug,
                                           **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_dataset_v3':
            dataset = TSEHeadTailDatasetV3(mode=mode, df=df,
                                           logger=self.logger,
                                           debug=self.debug,
                                           **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_dataset_v4':
            dataset = TSEHeadTailDatasetV4(mode=mode, df=df,
                                           logger=self.logger,
                                           debug=self.debug,
                                           **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_segmentation_dataset':
            dataset = TSEHeadTailSegmentationDataset(mode=mode, df=df,
                                                     logger=self.logger,
                                                     debug=self.debug,
                                                     **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_segmentation_dataset_v3':
            dataset = TSEHeadTailSegmentationDatasetV3(mode=mode, df=df,
                                                       logger=self.logger,
                                                       debug=self.debug,
                                                       **self.cfg_dataset)
        elif dataset_type == 'tse_headtail_segmentation_dataset_v4':
            dataset = TSEHeadTailSegmentationDatasetV4(mode=mode, df=df,
                                                       logger=self.logger,
                                                       debug=self.debug,
                                                       **self.cfg_dataset)
        else:
            raise NotImplementedError()

        if sampler_type == 'sequential':
            sampler = SequentialSampler(data_source=dataset)
        elif sampler_type == 'random':
            sampler = RandomSampler(data_source=dataset)
        elif sampler_type == 'weighted_random':
            is_neutrals = [
                row['sentiment'] == 'neutral' for i,
                row in df.iterrows()]
            weights = [
                neutral_weight if is_neutral else 1. for is_neutral in is_neutrals]
            is_longer_posnegs = [
                len(row['selected_text'].split()
                    ) > 20 and row['sentiment'] != 'neutral'
                for i, row in df.iterrows()]
            weights = [weight * longer_posneg_rate if is_longer_posneg else weight
                       for weight, is_longer_posneg in zip(weights, is_longer_posnegs)]
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights))
        else:
            raise NotImplementedError(
                f'sampler_type: {sampler_type} is not '
                'implemented for mode: {mode}')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=os.cpu_count(),
            # num_workers=1,
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    def _warmup(self, current_epoch, warmup_batch_or_epoch, model):
        module = model if self.device == 'cpu' else model.module
        if current_epoch == 0:
            for name, child in module.named_children():
                if 'classifier' in name:
                    self.logger.info(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    self.logger.info(name + ' is frozen')
                    for param in child.parameters():
                        param.requires_grad = False
        if current_epoch == warmup_batch_or_epoch:
            self.logger.info("Turn on all the layers")
            # for name, child in model.named_children():
            for name, child in module.named_children():
                for param in child.parameters():
                    param.requires_grad = True
        # for param in module.model.embeddings.parameters():
        #     param.requires_grad = False

    def _save_checkpoint(self, fold_num, current_epoch,
                         model, optimizer, scheduler,
                         val_textIDs, val_input_ids, val_preds, val_labels,
                         val_loss, best_thresh, best_jaccard):
        if not os.path.exists(f'./checkpoints/{self.exp_id}/{fold_num}'):
            os.makedirs(f'./checkpoints/{self.exp_id}/{fold_num}')
        # pth means pytorch
        cp_filename = f'./checkpoints/{self.exp_id}/{fold_num}/' \
            f'fold_{fold_num}_epoch_{current_epoch}_{val_loss:.5f}_{best_thresh:.5f}' \
            f'_{best_jaccard:.5f}_checkpoint.pth'
        # f'_{val_metric:.5f}_checkpoint.pth'
        module = model if self.device == 'cpu' else model.module
        cp_dict = {
            'fold_num': fold_num,
            'current_epoch': current_epoch,
            'model_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_textIDs': val_textIDs,
            'val_input_ids': val_input_ids,
            'val_preds': val_preds,
            'val_labels': val_labels,
            'histories': self.histories,
        }
        self.logger.info(f'now saving checkpoint to {cp_filename} ...')
        torch.save(cp_dict, cp_filename)

    def _search_best_filename(self, fold_num):
        # best_loss = np.inf
        best_metric = -1
        best_filename = ''
        for filename in glob(f'./checkpoints/{self.exp_id}/{fold_num}/*'):
            split_filename = filename.split('/')[-1].split('_')
            # temp_loss = float(split_filename[2])
            temp_metric = float(split_filename[6])
            # if temp_loss < best_loss:
            if temp_metric > best_metric:
                best_filename = filename
                # best_loss = temp_loss
                best_metric = temp_metric
        return best_filename  # , best_loss, best_acc

    def _load_best_checkpoint(self, fold_num):
        best_cp_filename = self._search_best_filename(fold_num)
        self.logger.info(f'the best file is {best_cp_filename} !')
        best_checkpoint = torch.load(best_cp_filename)
        return best_checkpoint


# class r001SegmentationRunner(Runner):
#     def __init__(self, exp_id, checkpoint, device, debug, config):
#         super().__init__(exp_id, checkpoint, device, debug, config,
#                          TSESegmentationDataset)
#
#     # def predict(self):
#     #     tst_ids = self._get_test_ids()
#     #     if self.debug:
#     #         tst_ids = tst_ids[:300]
#     #     test_loader = self._build_loader(
#     #         mode="test", ids=tst_ids, augment=None)
#     #     best_loss, best_acc = self._load_best_model()
#     #     test_ids, test_preds = self._test_loop(test_loader)
#
#     #     submission_df = pd.read_csv(
#     #         './mnt/inputs/origin/sample_submission.csv')
#     #     submission_df = submission_df.set_index('id_code')
#     #     submission_df.loc[test_ids, 'sirna'] = test_preds
#     #     submission_df = submission_df.reset_index()
#     #     filename_base = f'{self.exp_id}_{self.exp_time}_' \
#     #         f'{best_loss:.5f}_{best_acc:.5f}'
#     #     sub_filename = f'./mnt/submissions/{filename_base}_sub.csv'
#     #     submission_df.to_csv(sub_filename, index=False)
#
#     #     self.logger.info(f'Saved submission file to {sub_filename} !')
#     #     line_message = f'Finished the whole pipeline ! \n' \
#     #         f'Training time : {self.trn_time} min \n' \
#     #         f'Best valid loss : {best_loss:.5f} \n' \
#     #         f'Best valid acc : {best_acc:.5f}'
#     #     self.logger.send_line_notification(line_message)
#
#     def _train_loop(self, model, optimizer, fobj,
#                     loader, ema, use_special_mask):
#         model.train()
#         running_loss = 0
#
#         for batch in tqdm(loader):
#             input_ids = batch['input_ids'].to(self.device)
#             labels = batch['labels'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
#             special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                 if use_special_mask else None
#
#             (logits, ) = model(
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=attention_mask,
#                 special_tokens_mask=special_tokens_mask,
#             )
#
#             train_loss = fobj(logits, labels)
#
#             optimizer.zero_grad()
#             train_loss.backward()
#
#             optimizer.step()
#
#             running_loss += train_loss.item()
#
#             ema.on_batch_end(model)
#
#         train_loss = running_loss / len(loader)
#
#         return train_loss
#
#     def _valid_loop(self, model, fobj, loader, use_special_mask):
#         model.eval()
#         sigmoid = Sigmoid()
#         running_loss = 0
#
#         valid_textIDs_list = []
#         with torch.no_grad():
#             valid_textIDs, valid_input_ids, valid_preds, valid_labels \
#                 = [], [], [], []
#             for batch in tqdm(loader):
#                 textIDs = batch['textID']
#                 input_ids = batch['input_ids'].to(self.device)
#                 labels = batch['labels'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                     if use_special_mask else None
#
#                 (logits, ) = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     special_tokens_mask=special_tokens_mask,
#                 )
#
#                 valid_loss = fobj(logits, labels)
#                 running_loss += valid_loss.item()
#
#                 # _, predicted = torch.max(outputs.data, 1)
#                 predicted = sigmoid(logits.data)
#
#                 valid_textIDs_list.append(textIDs)
#                 valid_input_ids.append(input_ids.cpu())
#                 valid_preds.append(predicted.cpu())
#                 valid_labels.append(labels.cpu())
#
#             valid_loss = running_loss / len(loader)
#
#             valid_textIDs = list(
#                 itertools.chain.from_iterable(valid_textIDs_list))
#             valid_input_ids = torch.cat(valid_input_ids)
#             valid_preds = torch.cat(valid_preds)
#             valid_labels = torch.cat(valid_labels)
#
#             best_thresh, best_jaccard = \
#                 self._calc_jaccard(valid_input_ids,
#                                    valid_labels.bool(),
#                                    valid_preds,
#                                    loader.dataset.tokenizer,
#                                    self.cfg_train['thresh_unit'],
#                                    **self.cfg_predict)
#
#         return valid_loss, best_thresh, best_jaccard, valid_textIDs, \
#             valid_input_ids, valid_preds, valid_labels
#
#     # def _test_loop(self, loader):
#     #     self.model.eval()
#
#     #     test_ids = []
#     #     test_preds = []
#
#     #     sel_log('predicting ...', self.logger)
#     #     AUGNUM = 2
#     #     with torch.no_grad():
#     #         for (ids, images, labels) in tqdm(loader):
#     #             images, labels = images.to(
#     #                 self.device, dtype=torch.float), labels.to(
#     #                 self.device)
#     #             outputs = self.model.forward(images)
#     #             # avg predictions
#     #             # outputs = torch.mean(outputs.reshape((-1, 1108, 2)), 2)
#     #             # outputs = torch.mean(torch.stack(
#     #             #     [outputs[i::AUGNUM] for i in range(AUGNUM)], dim=2), dim=2)
#     #             # _, predicted = torch.max(outputs.data, 1)
#     #             sm_outputs = softmax(outputs, dim=1)
#     #             sm_outputs = torch.mean(torch.stack(
#     #                 [sm_outputs[i::AUGNUM] for i in range(AUGNUM)], dim=2), dim=2)
#     #             _, predicted = torch.max(sm_outputs.data, 1)
#
#     #             test_ids.append(ids[::2])
#     #             test_preds.append(predicted.cpu())
#
#     #         test_ids = np.concatenate(test_ids)
#     #         test_preds = torch.cat(test_preds).numpy()
#
#     #     return test_ids, test_preds
#
#     def _calc_jaccard(self, input_ids, selected_text_masks,
#                       y_preds, tokenizer, thresh_unit):
#         best_thresh = -1
#         best_jaccard = -1
#
#         self.logger.info('now calcurating the best threshold for jaccard ...')
#         for thresh in tqdm(list(np.arange(0.1, 1.0, thresh_unit))):
#             # get predicted texts
#             predicted_text_masks = [y_pred > thresh for y_pred in y_preds]
#             # calc jaccard for this threshold
#             temp_jaccard = 0
#             for input_id, selected_text_mask, predicted_text_mask in zip(
#                     input_ids, selected_text_masks, predicted_text_masks):
#                 selected_text = tokenizer.decode(
#                     input_id[selected_text_mask])
#                 # fill continuous zeros between one
#                 _non_zeros = predicted_text_mask.nonzero()
#                 if _non_zeros.shape[0] > 0:
#                     _predicted_text_mask_min = _non_zeros.min()
#                     _predicted_text_mask_max = _non_zeros.max()
#                     predicted_text_mask[_predicted_text_mask_min:
#                                         _predicted_text_mask_max + 1] = True
#                 predicted_text = tokenizer.decode(
#                     input_id[predicted_text_mask])
#                 temp_jaccard += jaccard(selected_text, predicted_text)
#
#             temp_jaccard /= len(selected_text_masks)
#             # update the best jaccard
#             if temp_jaccard > best_jaccard:
#                 best_thresh = thresh
#                 best_jaccard = temp_jaccard
#
#         assert best_thresh != -1
#         assert best_jaccard != -1
#
#         return best_thresh, best_jaccard


class r002HeadTailRunner(Runner):
    def predict(self, tst_filename, checkpoints):
        fold_test_preds_heads, fold_test_preds_tails = [], []
        for checkpoint in checkpoints:
            # load and preprocess train.csv
            tst_df = pd.read_csv(tst_filename)
            if self.cfg_invalid_labels:
                tst_df = tst_df.set_index('textID')
                for invalid_label_csv in self.cfg_invalid_labels:
                    invalid_label_df = pd.read_csv(invalid_label_csv)
                    for i, row in invalid_label_df.iterrows():
                        tst_df.loc[row['textID'], 'selected_text'] = \
                            row['guchio_selected_text']
                tst_df = tst_df.reset_index()

            # load and apply checkpoint if needed
            if checkpoint:
                checkpoint = torch.load(checkpoint)
            else:
                raise Exception('predict needs checkpoint')

            tst_loader = self._build_loader(mode='test', df=tst_df,
                                            **self.cfg_loader)

            # build model and related objects
            # these objects have state
            model = self._get_model(**self.cfg_model)
            module = model if self.device == 'cpu' else model.module
            module.resize_token_embeddings(
                len(tst_loader.dataset.tokenizer))  # for sentiment

            if checkpoint:
                module.load_state_dict(checkpoint['model_state_dict'])

            model = model.to(self.device)

            use_special_mask = self.cfg_train['use_special_mask']
            use_offsets = self.cfg_predict['use_offsets']
            textIDs, test_texts, test_input_ids, test_offsets, \
                test_sentiments, test_preds_head, test_preds_tail\
                = self._test_loop(model, tst_loader,
                                  use_special_mask, use_offsets)

            fold_test_preds_heads.append(test_preds_head)
            fold_test_preds_tails.append(test_preds_tail)

        avg_test_preds_head = torch.mean(
            torch.stack(fold_test_preds_heads), dim=0)
        avg_test_preds_tail = torch.mean(
            torch.stack(fold_test_preds_tails), dim=0)
        if use_offsets:
            predicted_texts = self._get_predicted_texts_offsets(
                test_texts,
                test_offsets,
                test_sentiments,
                avg_test_preds_head,
                avg_test_preds_tail,
                self.cfg_predict['neutral_origin'],
                self.cfg_predict['head_tail_equal_handle'],
                self.cfg_predict['pospro'],
                self.cfg_predict['tail_index'],
            )
        else:
            predicted_texts = self._get_predicted_texts(
                test_texts,
                test_input_ids,
                test_sentiments,
                avg_test_preds_head,
                avg_test_preds_tail,
                tst_loader.dataset.tokenizer,
                self.cfg_predict['neutral_origin'],
                self.cfg_predict['head_tail_equal_handle'],
                self.cfg_predict['pospro'],
                self.cfg_predict['tail_index']
            )

        return textIDs, predicted_texts

    def _mk_char_preds(self, offsets, preds_head, preds_tail):
        char_preds_heads, char_preds_tails = [], []
        # 最初の４つは無視
        for offset, pred_head, pred_tail in tqdm(zip(offsets[4:], preds_head[4:], preds_tail[4:])):
            for offset_i, pred_head_i, pred_tail_i in tqdm(zip(offset, pred_head, pred_tail)):
                char_preds_head, char_preds_tail = np.zeros(141), np.zeros(141)
                char_preds_head[offset_i[0]:offset_i[1]] = pred_head_i
                char_preds_tail[offset_i[0]:offset_i[1]] = pred_tail_i
                char_preds_heads.append(char_preds_head)
                char_preds_tails.append(char_preds_tail)
        char_preds_heads, char_preds_tails = np.asarray(char_preds_heads), np.asarray(char_preds_tails)
        return char_preds_heads, char_preds_tails

    def predict_proba(self, tst_df, checkpoints):
        avg_test_char_preds_head, avg_test_char_preds_tail = None, None
        for checkpoint in checkpoints:
            # load and preprocess train.csv
            if self.cfg_invalid_labels:
                tst_df = tst_df.set_index('textID')
                for invalid_label_csv in self.cfg_invalid_labels:
                    invalid_label_df = pd.read_csv(invalid_label_csv)
                    for i, row in invalid_label_df.iterrows():
                        tst_df.loc[row['textID'], 'selected_text'] = \
                            row['guchio_selected_text']
                tst_df = tst_df.reset_index()

            # load and apply checkpoint if needed
            if checkpoint:
                checkpoint = torch.load(checkpoint)
            else:
                raise Exception('predict needs checkpoint')

            tst_loader = self._build_loader(mode='test', df=tst_df,
                                            **self.cfg_loader)

            # build model and related objects
            # these objects have state
            model = self._get_model(**self.cfg_model)
            module = model if self.device == 'cpu' else model.module
            module.resize_token_embeddings(
                len(tst_loader.dataset.tokenizer))  # for sentiment

            if checkpoint:
                module.load_state_dict(checkpoint['model_state_dict'])

            model = model.to(self.device)

            use_special_mask = self.cfg_train['use_special_mask']
            use_offsets = self.cfg_predict['use_offsets']
            textIDs, test_texts, test_input_ids, test_offsets, \
                test_sentiments, test_preds_head, test_preds_tail\
                = self._test_loop(model, tst_loader,
                                  use_special_mask, use_offsets)

            test_char_preds_head, test_char_preds_tail = self._mk_char_preds(test_offsets, test_preds_head, test_preds_tail)

            if avg_test_char_preds_head is None:
                avg_test_char_preds_head = test_char_preds_head / len(checkpoints)
            else:
                avg_test_char_preds_head += test_char_preds_head / len(checkpoints)
            if avg_test_char_preds_tail is None:
                avg_test_char_preds_tail = test_char_preds_tail / len(checkpoints)
            else:
                avg_test_char_preds_tail += test_char_preds_tail / len(checkpoints)

            del checkpoint, test_char_preds_head, test_char_preds_tail
            gc.collect()

        # avg_test_char_preds_head = np.mean(
        #     np.stack(fold_test_char_preds_heads), axis=0)
        # avg_test_char_preds_tail = np.mean(
        #     np.stack(fold_test_char_preds_tails), axis=0)
        # if use_offsets:
        #     predicted_texts = self._get_predicted_texts_offsets(
        #         test_texts,
        #         test_offsets,
        #         test_sentiments,
        #         avg_test_preds_head,
        #         avg_test_preds_tail,
        #         self.cfg_predict['neutral_origin'],
        #         self.cfg_predict['head_tail_equal_handle'],
        #         self.cfg_predict['pospro'],
        #         self.cfg_predict['tail_index'],
        #     )
        # else:
        #     predicted_texts = self._get_predicted_texts(
        #         test_texts,
        #         test_input_ids,
        #         test_sentiments,
        #         avg_test_preds_head,
        #         avg_test_preds_tail,
        #         tst_loader.dataset.tokenizer,
        #         self.cfg_predict['neutral_origin'],
        #         self.cfg_predict['head_tail_equal_handle'],
        #         self.cfg_predict['pospro'],
        #         self.cfg_predict['tail_index']
        #     )

        return (avg_test_char_preds_head, avg_test_char_preds_tail)



    def _train_loop(self, model, optimizer, fobj,
                    loader, warmup_batch, ema, accum_mod, use_specical_mask,
                    fobj_segmentation, segmentation_loss_ratio,
                    loss_weight_type, fobj_index_diff, use_dist_loss,
                    single_word):
        model.train()
        running_loss = 0

        softargmax1d = SoftArgmax1D(
            beta=5., device=self.device).to(
            self.device)
        ce_loss = CrossEntropyLoss()

        for batch_i, batch in enumerate(tqdm(loader)):
            if warmup_batch > 0:
                self._warmup(batch_i, warmup_batch, model)

            input_ids = batch['input_ids'].to(self.device)
            labels_head = batch['labels_head'].to(self.device)
            labels_tail = batch['labels_tail'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
                if use_specical_mask else None

            (logits, ) = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )

            # 5 is temerature
            logits_head = logits[0]
            logits_tail = logits[1]

            if loss_weight_type == 'sel_len':
                sel_len_weight = 1. * (
                    1. / (labels_tail - labels_head).float())
                train_losses_head = fobj(logits_head, labels_head)
                train_loss = (train_losses_head * sel_len_weight).mean()
                train_losses_tail = fobj(logits_tail, labels_tail)
                train_loss += (train_losses_tail * sel_len_weight).mean()
            elif loss_weight_type == 'sel_len_log':
                sel_len_weight = 1. * (
                    1. / (labels_tail - labels_head).float() / 10. + 2.71828).log()
                # 1. / (labels_tail - labels_head).float() + 2.71828).log()
                train_losses_head = fobj(logits_head, labels_head)
                train_loss = (train_losses_head * sel_len_weight).mean()
                train_losses_tail = fobj(logits_tail, labels_tail)
                train_loss += (train_losses_tail * sel_len_weight).mean()
            else:
                # train_loss = fobj(logits_head, labels_head)
                # train_loss += fobj(logits_tail, labels_tail)
                train_loss = self.cfg_train['head_ratio'] * \
                    fobj(logits_head, labels_head)
                train_loss += self.cfg_train['tail_ratio'] * \
                    fobj(logits_tail, labels_tail)

            if fobj_segmentation:
                labels_segmentation = batch['labels_segmentation']\
                    .to(self.device)
                logits_segmentation = logits[2]
                # logits_segmentation *= attention_mask

                if self.cfg_fobj_segmentation['fobj_type'] == 'lovasz':
                    train_loss += segmentation_loss_ratio * \
                        fobj_segmentation(logits_segmentation,
                                          labels_segmentation,
                                          ignore=-1)
                else:
                    train_loss += segmentation_loss_ratio * \
                        fobj_segmentation(logits_segmentation,
                                          labels_segmentation)

            if single_word:
                labels_single = batch['labels_single_word'].long()\
                    .to(self.device)
                logits_single = logits[3]
                train_loss += ce_loss(logits_single, labels_single)

            if fobj_index_diff:
                pred_index_head = softargmax1d(logits_head)
                pred_index_tail = softargmax1d(logits_tail)
                pred_index_diff = pred_index_tail - pred_index_head
                labels_index_diff = (labels_tail - labels_head).float()
                train_loss += 0.0003 * fobj_index_diff(pred_index_diff,
                                                       labels_index_diff)
                # train_loss += 0.003 * fobj_index_diff(pred_index_head,
                #                                       labels_head.float())
                # train_loss += 0.003 * fobj_index_diff(pred_index_tail,
                #                                       labels_tail.float())

            if use_dist_loss:
                # train_loss += dist_loss(logits_head, logits_tail,
                #                         labels_tail, labels_tail,
                # self.device, loader.dataset.max_length)
                pred_index_head = softargmax1d(logits_head)
                pred_index_tail = softargmax1d(logits_tail)
                pred_index_diff = (
                    pred_index_tail - pred_index_head) / loader.dataset.max_length
                labels_index_diff = (
                    (labels_tail - labels_head).float()) / loader.dataset.max_length
                diff = (pred_index_diff - labels_index_diff).mean()
                train_loss += -torch.log(1 - torch.sqrt(diff * diff))

            train_loss.backward()

            running_loss += train_loss.item()

            if (batch_i + 1) % accum_mod == 0:
                optimizer.step()
                optimizer.zero_grad()

                ema.on_batch_end(model)

        train_loss = running_loss / len(loader)

        return train_loss

    def _valid_loop(self, model, fobj, loader, use_special_mask,
                    use_offsets, loss_weight_type, single_word):
        model.eval()
        softmax = Softmax(dim=1)
        running_loss = 0

        valid_textIDs_list = []
        with torch.no_grad():
            valid_texts = []
            valid_textIDs = []
            valid_input_ids = []
            valid_offsets = []
            valid_sentiments = []
            valid_preds_single = []
            valid_selected_texts = []
            valid_preds_head, valid_preds_tail = [], []
            valid_labels_head, valid_labels_tail = [], []
            for batch in tqdm(loader):
                textIDs = batch['textID']
                valid_text = batch['text']
                input_ids = batch['input_ids'].to(self.device)
                if use_offsets:
                    valid_offset = batch['offsets'].to(self.device)
                valid_sentiment = batch['sentiment']
                selected_texts = batch['selected_text']
                labels_head = batch['labels_head'].to(self.device)
                labels_tail = batch['labels_tail'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
                    if use_special_mask else None

                (logits, ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                )
                logits_head = logits[0]
                logits_tail = logits[1]
                if single_word:
                    predicted_single = softmax(logits[3].data)
                    valid_preds_single.append(predicted_single)
                else:
                    valid_preds_single.append(torch.zeros(logits_head.shape))

                if loss_weight_type == 'sel_len':
                    sel_len_weight = 1. * (
                        1. / (labels_tail - labels_head).float() + 1.)
                    valid_losses_head = fobj(logits_head, labels_head)
                    valid_loss = (valid_losses_head * sel_len_weight).mean()
                    valid_losses_tail = fobj(logits_tail, labels_tail)
                    valid_loss += (valid_losses_tail * sel_len_weight).mean()
                elif loss_weight_type == 'sel_len_log':
                    sel_len_weight = 1. * (
                        1. / (labels_tail - labels_head).float() / 10. + 2.71828).log()
                    valid_losses_head = fobj(logits_head, labels_head)
                    valid_loss = (valid_losses_head * sel_len_weight).mean()
                    valid_losses_tail = fobj(logits_tail, labels_tail)
                    valid_loss += (valid_losses_tail * sel_len_weight).mean()
                else:
                    valid_loss = fobj(logits_head, labels_head)
                    valid_loss += fobj(logits_tail, labels_tail)
                running_loss += valid_loss.item()

                # _, predicted = torch.max(outputs.data, 1)
                predicted_head = softmax(logits_head.data)
                predicted_tail = softmax(logits_tail.data)

                valid_textIDs_list.append(textIDs)
                valid_texts.append(valid_text)
                valid_input_ids.append(input_ids.cpu())
                if use_offsets:
                    valid_offsets.append(valid_offset.cpu())
                valid_sentiments.append(valid_sentiment)
                valid_selected_texts.append(selected_texts)
                valid_preds_head.append(predicted_head.cpu())
                valid_preds_tail.append(predicted_tail.cpu())
                valid_labels_head.append(labels_head.cpu())
                valid_labels_tail.append(labels_tail.cpu())

            valid_loss = running_loss / len(loader)

            valid_textIDs = list(
                itertools.chain.from_iterable(valid_textIDs_list))
            valid_texts = list(itertools.chain.from_iterable(valid_texts))
            valid_input_ids = torch.cat(valid_input_ids)
            if use_offsets:
                valid_offsets = torch.cat(valid_offsets)
            valid_sentiments = list(
                itertools.chain.from_iterable(valid_sentiments))
            valid_selected_texts = list(
                itertools.chain.from_iterable(valid_selected_texts))
            valid_preds_head = torch.cat(valid_preds_head)
            valid_preds_tail = torch.cat(valid_preds_tail)
            if single_word:
                valid_preds_single = torch.cat(valid_preds_single)
            else:
                valid_preds_single = list(
                    itertools.chain.from_iterable(valid_preds_single))
            valid_labels_head = torch.cat(valid_labels_head)
            valid_labels_tail = torch.cat(valid_labels_tail)

            if use_offsets:
                best_thresh, best_jaccard = \
                    self._calc_jaccard_offsets(valid_texts,
                                               valid_offsets,
                                               valid_sentiments,
                                               valid_selected_texts,
                                               valid_preds_head,
                                               valid_preds_tail,
                                               self.cfg_predict['neutral_origin'],
                                               self.cfg_predict['head_tail_equal_handle'],
                                               self.cfg_predict['pospro'],
                                               self.cfg_predict['tail_index'],
                                               )
            else:
                best_thresh, best_jaccard = \
                    self._calc_jaccard(valid_texts,
                                       valid_input_ids,
                                       valid_sentiments,
                                       valid_selected_texts,
                                       valid_preds_head,
                                       valid_preds_tail,
                                       valid_preds_single,
                                       loader.dataset.tokenizer,
                                       self.cfg_train['thresh_unit'],
                                       self.cfg_predict['neutral_origin'],
                                       self.cfg_predict['head_tail_equal_handle'],
                                       self.cfg_predict['pospro'],
                                       self.cfg_predict['tail_index'],
                                       )

        valid_preds = (valid_preds_head, valid_preds_tail)
        valid_labels = (valid_labels_head, valid_labels_tail)

        return valid_loss, best_thresh, best_jaccard, valid_textIDs, \
            valid_input_ids, valid_preds, valid_labels

    # def _calc_jaccard(self, input_ids, labels_head, labels_tail,
    #                  y_preds_head, y_preds_tail, tokenizer, thresh_unit):

    #    temp_jaccard = 0
    #    for input_id, label_head, label_tail, y_pred_head, y_pred_tail \
    #            in zip(input_ids, labels_head, labels_tail,
    #                   y_preds_head, y_preds_tail):
    #        selected_text = tokenizer.decode(
    #            input_id[label_head:label_tail])
    #        pred_label_head = y_pred_head.argmax()
    #        pred_label_tail = y_pred_tail.argmax()
    #        predicted_text = tokenizer.decode(
    #            input_id[pred_label_head:pred_label_tail])
    #        temp_jaccard += jaccard(selected_text, predicted_text)

    #    best_thresh = -1
    #    best_jaccard = temp_jaccard / len(input_ids)

    #    return best_thresh, best_jaccard

    def _get_predicted_texts(self, texts, input_ids, sentiments, y_preds_head,
                             y_preds_tail, y_preds_single, tokenizer,
                             neutral_origin=False,
                             head_tail_equal_handle='tail',
                             pospro={},
                             tail_index='natural'):
        predicted_texts = []
        for text, input_id, sentiment, y_pred_head, y_pred_tail, y_pred_single \
                in zip(texts, input_ids, sentiments, y_preds_head, y_preds_tail, y_preds_single):
            if neutral_origin and sentiment == 'neutral':
                predicted_texts.append(text)
                continue
            if y_pred_single.sum() > 0.5 and y_pred_single.argmax() != 0:
                predicted_texts.append(tokenizer.decode(
                    [input_id[y_pred_single.argmax()]]))
                continue
            if pospro['head_tail_1']:
                pred_label_head, pred_label_tail = self.calc_best_se_indexes(
                    y_pred_head, y_pred_tail)
            else:
                pred_label_head = y_pred_head.argmax()
                pred_label_tail = y_pred_tail.argmax()

            if tail_index == 'kernel':
                pred_label_tail += 1

            # if pred_label_head > pred_label_tail or len(text.split()) < 2:
            #     predicted_text = text
            if pred_label_head >= pred_label_tail:
                if head_tail_equal_handle == 'nothing':
                    predicted_text = ''
                elif head_tail_equal_handle == 'head':
                    predicted_text = tokenizer.decode(
                        input_id[pred_label_head:pred_label_tail + 1])
                elif head_tail_equal_handle == 'tail':
                    predicted_text = tokenizer.decode(
                        input_id[pred_label_head - 1:pred_label_tail])
                elif head_tail_equal_handle == 'larger':
                    while pred_label_head >= pred_label_tail:
                        print(f'flip found, {text[:10]}...')
                        if y_pred_head.max() <= y_pred_tail.max():
                            # NOTE: copy にしたい
                            y_pred_head[y_pred_head.argmax()] = 0.
                        else:
                            y_pred_tail[y_pred_tail.argmax()] = 0.
                        pred_label_head = y_pred_head.argmax()
                        pred_label_tail = y_pred_tail.argmax()
                        if tail_index == 'kernel':
                            pred_label_tail += 1
                    predicted_text = tokenizer.decode(
                        input_id[pred_label_head:pred_label_tail])
                elif head_tail_equal_handle == 'larger_2':
                    if y_pred_head.max() <= y_pred_tail.max():
                        y_pred_head = y_pred_tail - 1
                    else:
                        y_pred_tail = y_pred_head + 1
                    predicted_text = tokenizer.decode(
                        input_id[pred_label_head:pred_label_tail])
                else:
                    raise NotImplementedError()
            else:
                predicted_text = tokenizer.decode(
                    input_id[pred_label_head:pred_label_tail])

            if self.cfg_dataset['tokenize_period']:
                predicted_text = re.sub(r'\[S\]', ' ', predicted_text)
                predicted_text = re.sub(r'\[PERIOD\]', '.', predicted_text)
                predicted_text = re.sub(r'\[EXCL\]', '!', predicted_text)
                predicted_text = re.sub(r'\[QUES\]', '?', predicted_text)

            if pospro['req_shorten']:
                if len(predicted_text.split()) == 1:
                    predicted_text = predicted_text.replace('!!!!', '!')
                    predicted_text = predicted_text.replace('..', '.')
                    predicted_text = predicted_text.replace('...', '.')
            if pospro['regex_1']:
                a = re.findall('[^A-Za-z0-9]', predicted_text)
                b = re.sub('[^A-Za-z0-9]+', '', predicted_text)
                try:
                    if a.count('.') == 3:
                        predicted_text = b + '. ' + b + '..'
                    elif a.count('!') == 4:
                        predicted_text = b + '! ' + b + '!! ' + b + '!!!'
                    else:
                        predicted_text = predicted_text
                except BaseException:
                    predicted_text = predicted_text
            if pospro['regex_2']:
                predicted_text = re.sub(r'^(\.+)', '.', predicted_text)
            if pospro['regex_3']:
                predicted_text = re.sub(r'^(\.+)', '.', predicted_text)
                predicted_text = re.sub('^(!+)', '!', predicted_text)
                if len(predicted_text.split()) == 1:
                    predicted_text = re.sub(
                        r'\.\.\.\.\.\.\.$', '..', predicted_text)
                    predicted_text = re.sub(
                        r'\.\.\.\.\.\.$', '..', predicted_text)
                    predicted_text = re.sub(
                        r'\.\.\.\.\.$', '.', predicted_text)
                    predicted_text = re.sub(r'\.\.\.\.$', '..', predicted_text)
                    predicted_text = re.sub(r'\.\.\.$', '..', predicted_text)
                    predicted_text = re.sub(r'\.\.\.$', '..', predicted_text)
                    predicted_text = re.sub('!!!!!!!!$', '!', predicted_text)
                    predicted_text = re.sub('!!!!!$', '!', predicted_text)
                    predicted_text = re.sub('!!!!$', '!', predicted_text)
                    predicted_text = re.sub('!!!$', '!!', predicted_text)
            predicted_texts.append(predicted_text)

        return predicted_texts

    def calc_best_se_indexes(self, _start_logits, _end_logits):
        best_logit = -1000
        best_idxs = None
        for start_idx, start_logit in enumerate(tqdm(_start_logits)):
            for end_idx, end_logit in enumerate(_end_logits[start_idx:]):
                logit_sum = (start_logit + end_logit).item()
                if logit_sum > best_logit:
                    best_logit = logit_sum
                    best_idxs = (start_idx, start_idx + end_idx)
        return best_idxs

    def _calc_jaccard(self, texts, input_ids, sentiments, selected_texts,
                      y_preds_head, y_preds_tail, y_preds_single, tokenizer, thresh_unit,
                      neutral_origin=False, head_tail_equal_handle='tail',
                      pospro={}, tail_index='natural'):

        temp_jaccard = 0
        predicted_texts = self._get_predicted_texts(
            texts,
            input_ids,
            sentiments,
            y_preds_head,
            y_preds_tail,
            y_preds_single,
            tokenizer,
            neutral_origin,
            head_tail_equal_handle,
            pospro,
            tail_index
        )
        for selected_text, predicted_text in zip(
                selected_texts, predicted_texts):
            temp_jaccard += jaccard(selected_text, predicted_text)
            if ('.' in selected_text or '.' in predicted_text) and jaccard(
                    selected_text, predicted_text) == 0.:
                print('---------------')
                print(
                    f'selected_text: {selected_text} -- predicted_text: {predicted_text}')

        best_thresh = -1
        best_jaccard = temp_jaccard / len(input_ids)

        return best_thresh, best_jaccard

    def modify_punc_length(self, text, selected_text):
        last_char_punc = True  # とりあえず探索
        x = -1  # 末尾から探す
        while last_char_punc:
            if abs(x) > len(selected_text):
                self.logger.warn(
                    f'x is longer than selected_text, {x}: {selected_text}')
                return selected_text
            if selected_text[x] not in ("!", ".", "?"):
                last_char_punc = False
                break
            else:
                x -= 1
        conti_punc = abs(x) - 1  # 末尾から連続する数
        if conti_punc >= 3:
            selected_text = selected_text[:-(conti_punc - 2)]
        elif conti_punc == 2:
            pass  # 何もしなくていい
        elif conti_punc == 1:  # 元のtextを探しに行く
            f_idx0 = text.find(selected_text)
            f_idx1 = f_idx0 + len(selected_text) - 1
            if f_idx1 + 1 == len(text):
                pass
            else:
                if text[f_idx1 + 1] in ("!", ".", "?"):
                    f_idx1 += 1
                selected_text = text[f_idx0:f_idx1 + 1]
        else:
            pass
        return selected_text

    def _get_predicted_texts_offsets(self, texts, offsets_list, sentiments,
                                     y_preds_head, y_preds_tail,
                                     neutral_origin=False,
                                     head_tail_equal_handle='tail',
                                     pospro={},
                                     tail_index='neutral'):
        predicted_texts = []
        for text, offsets, sentiment, y_pred_head, y_pred_tail \
                in zip(texts, offsets_list, sentiments,
                       y_preds_head, y_preds_tail):
            if neutral_origin and sentiment == 'neutral' \
                    or len(text.split()) < 2:
                predicted_texts.append(text)
                continue
            text1 = ' ' + ' '.join(text.split())
            pred_label_head = y_pred_head.argmax()
            pred_label_tail = y_pred_tail.argmax()

            if tail_index == 'kernel':
                pred_label_tail += 1

            # NOTE: change here from kernel
            if pred_label_head >= pred_label_tail:
                if head_tail_equal_handle == 'nothing':
                    raise NotImplementedError()
                elif head_tail_equal_handle == 'head':
                    pred_label_tail = pred_label_head + 1
                elif head_tail_equal_handle == 'tail':
                    pred_label_head = pred_label_tail - 1
                elif head_tail_equal_handle == 'larger':
                    while pred_label_head >= pred_label_tail:
                        print(f'flip found, {text[:10]}...')
                        if y_pred_head.max() <= y_pred_tail.max():
                            # NOTE: copy にしたい
                            y_pred_head[y_pred_head.argmax()] = 0.
                        else:
                            y_pred_tail[y_pred_tail.argmax()] = 0.
                        pred_label_head = y_pred_head.argmax()
                        pred_label_tail = y_pred_tail.argmax()
                        if tail_index == 'kernel':
                            pred_label_tail += 1
                elif head_tail_equal_handle == 'larger_2':
                    if y_pred_head.max() <= y_pred_tail.max():
                        y_pred_head = y_pred_tail - 1
                    else:
                        y_pred_tail = y_pred_head + 1
                else:
                    raise NotImplementedError()

            # predicted_text = ''
            # for ix in range(pred_label_head, pred_label_tail):
            #     predicted_text += text[offsets[ix][0]:offsets[ix][1]]
            #     if (ix + 1) < len(offsets) and \
            #             offsets[ix][1] < offsets[ix + 1][0]:
            #         predicted_text += ' '
            ss = offsets[pred_label_head][0]
            ee = offsets[pred_label_tail - 1][1]
            predicted_text = text1[ss:ee].strip()
            if pospro['head_tail_1']:
                raise NotImplementedError()
            if pospro['req_shorten']:
                raise NotImplementedError()
            if pospro['regex_1']:
                raise NotImplementedError()
            if pospro['regex_2']:
                raise NotImplementedError()
            if pospro['regex_3']:
                raise NotImplementedError()
            if pospro['magic']:
                ee += 1
                ee -= text[ss:ee].strip().count('   ')
                ee += text[ss:ee].strip().count('  ')
                if '  ' in text[:(ss + ee) // 2]:
                    predicted_text = text[ss:ee].strip()
            if pospro['magic_2']:
                original_text = text
                y_start_char = ss
                y_end_char = ee
                y_selected_text = text1[y_start_char:y_end_char].strip()

                if (len(y_selected_text) > 1 and y_end_char < len(text1) and
                    y_selected_text[-1] == '.' and
                    (text1[y_end_char] == '.' or
                     y_selected_text[-2] == '.')):
                    y_selected_text = re.sub(r'\.+$', '..', y_selected_text)
                tmp = re.sub(
                    r"([\\\*\+\.\?\{\}\(\)\[\]\^\$\|])",
                    r"\\\g<0>",
                    y_selected_text)
                tmp = re.sub(r" ", " +", tmp)
                m = re.search(tmp, original_text)
                ss2 = m.start()
                ee2 = m.end()
                if '  ' in original_text[:(ss2 + ee2) // 2]:
                    ss = y_start_char
                    ee = y_end_char + 1
                    if sentiment == 'neutral':
                        ee += 1
                    st = original_text[ss:ee].strip(' ')
                    # re.sub(r' .$', '', st).strip('`')  # この一行追加
                    y_selected_text = st
                else:
                    if (ee2 < len(original_text) -
                            1 and original_text[ee2:ee2 + 2] in ('..', '!!', '??', '((', '))')):
                        ee2 += 1
                    # 先頭の空白分後退
                    if original_text[0] == ' ':
                        ss2 -= 1
                    y_selected_text = original_text[ss2:ee2].strip(' ½')
                if text1[:ee2 + 5] == " " + text[:ee2 +
                                                 4] and sentiment != 'neutral':  # 簡単のため、長さが同じ場合に限定している
                    y_selected_text = self.modify_punc_length(
                        text, y_selected_text)
                predicted_text = y_selected_text

            predicted_texts.append(predicted_text)

        return predicted_texts

    def _calc_jaccard_offsets(self, texts, offsets_list,
                              sentiments, selected_texts,
                              y_preds_head, y_preds_tail,
                              neutral_origin=False,
                              head_tail_equal_handle='tail',
                              pospro={},
                              tail_index='neutral',
                              ):
        temp_jaccard = 0
        predicted_texts = self._get_predicted_texts_offsets(
            texts,
            offsets_list,
            sentiments,
            y_preds_head,
            y_preds_tail,
            neutral_origin,
            head_tail_equal_handle,
            pospro,
            tail_index
        )
        for selected_text, predicted_text in zip(
                selected_texts, predicted_texts):
            temp_jaccard += jaccard(selected_text, predicted_text)
            # if jaccard(selected_text, predicted_text) != 1.:
            #     print('--------------')
            #     print(f'selected_text: {selected_text}')
            #     print(f'predicted_text: {predicted_text}')

        best_thresh = -1
        best_jaccard = temp_jaccard / len(texts)

        return best_thresh, best_jaccard

    def _test_loop(self, model, loader, use_special_mask, use_offsets):
        model.eval()
        softmax = Softmax(dim=1)

        with torch.no_grad():
            textIDs = []
            test_texts = []
            test_input_ids = []
            test_offsets = []
            test_sentiments = []
            test_preds_head = []
            test_preds_tail = []

            for batch in tqdm(loader):
                textID = batch['textID']
                test_text = batch['text']
                input_ids = batch['input_ids'].to(self.device)
                if use_offsets:
                    offsets = batch['offsets'].to(self.device)
                else:
                    offsets = None
                sentiment = batch['sentiment']
                attention_mask = batch['attention_mask'].to(self.device)
                special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
                    if use_special_mask else None

                (logits, ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                )
                logits_head = logits[0]
                logits_tail = logits[1]

                predicted_head = softmax(logits_head.data)
                predicted_tail = softmax(logits_tail.data)

                test_texts.append(test_text)
                textIDs.append(textID)
                test_input_ids.append(input_ids.cpu())
                if use_offsets:
                    test_offsets.append(offsets.cpu())
                test_sentiments.append(sentiment)
                test_preds_head.append(predicted_head.cpu())
                test_preds_tail.append(predicted_tail.cpu())

            test_texts = list(chain.from_iterable(test_texts))
            textIDs = list(chain.from_iterable(textIDs))
            test_input_ids = torch.cat(test_input_ids)
            if use_offsets:
                test_offsets = torch.cat(test_offsets)
            test_sentiments = list(chain.from_iterable(test_sentiments))
            test_preds_head = torch.cat(test_preds_head)
            test_preds_tail = torch.cat(test_preds_tail)

        return textIDs, test_texts, test_input_ids, test_offsets, \
            test_sentiments, test_preds_head, test_preds_tail


# class r003HeadTailSegmentRunner(r002HeadTailRunner):
#
#     def _valid_loop(self, model, fobj, loader, use_special_mask, use_offsets):
#         model.eval()
#         # softmax = Softmax(dim=1)
#         sigmoid = Sigmoid()
#         running_loss = 0
#
#         valid_textIDs_list = []
#         with torch.no_grad():
#             valid_texts = []
#             valid_textIDs = []
#             valid_input_ids = []
#             valid_offsets = []
#             valid_sentiments = []
#             valid_selected_texts = []
#             valid_preds_head, valid_preds_tail = [], []
#             valid_labels_head, valid_labels_tail = [], []
#             for batch in tqdm(loader):
#                 textIDs = batch['textID']
#                 valid_text = batch['text']
#                 input_ids = batch['input_ids'].to(self.device)
#                 if use_offsets:
#                     valid_offset = batch['offsets'].to(self.device)
#                 valid_sentiment = batch['sentiment']
#                 selected_texts = batch['selected_text']
#                 labels_head = batch['labels_head'].to(self.device)
#                 labels_tail = batch['labels_tail'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                     if use_special_mask else None
#
#                 (logits, ) = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     special_tokens_mask=special_tokens_mask,
#                 )
#                 logits_head = logits[0]
#                 logits_tail = logits[1]
#
#                 valid_loss = fobj(logits_head, labels_head)
#                 valid_loss += fobj(logits_tail, labels_tail)
#                 running_loss += valid_loss.item()
#
#                 # _, predicted = torch.max(outputs.data, 1)
#                 predicted_head = sigmoid(logits_head.data)
#                 predicted_tail = sigmoid(logits_tail.data)
#
#                 valid_textIDs_list.append(textIDs)
#                 valid_texts.append(valid_text)
#                 valid_input_ids.append(input_ids.cpu())
#                 if use_offsets:
#                     valid_offsets.append(valid_offset.cpu())
#                 valid_sentiments.append(valid_sentiment)
#                 valid_selected_texts.append(selected_texts)
#                 valid_preds_head.append(predicted_head.cpu())
#                 valid_preds_tail.append(predicted_tail.cpu())
#                 valid_labels_head.append(labels_head.cpu())
#                 valid_labels_tail.append(labels_tail.cpu())
#
#             valid_loss = running_loss / len(loader)
#
#             valid_textIDs = list(
#                 itertools.chain.from_iterable(valid_textIDs_list))
#             valid_texts = list(itertools.chain.from_iterable(valid_texts))
#             valid_input_ids = torch.cat(valid_input_ids)
#             if use_offsets:
#                 valid_offsets = torch.cat(valid_offsets)
#             valid_sentiments = list(
#                 itertools.chain.from_iterable(valid_sentiments))
#             valid_selected_texts = list(
#                 itertools.chain.from_iterable(valid_selected_texts))
#             valid_preds_head = torch.cat(valid_preds_head)
#             valid_preds_tail = torch.cat(valid_preds_tail)
#             valid_labels_head = torch.cat(valid_labels_head)
#             valid_labels_tail = torch.cat(valid_labels_tail)
#
#             if use_offsets:
#                 best_thresh, best_jaccard = \
#                     self._calc_jaccard_offsets(valid_texts,
#                                                valid_offsets,
#                                                valid_sentiments,
#                                                valid_selected_texts,
#                                                valid_preds_head,
#                                                valid_preds_tail,
#                                                self.cfg_predict['neutral_origin'],
#                                                self.cfg_predict['head_tail_equal_handle'],
#                                                )
#             else:
#                 best_thresh, best_jaccard = \
#                     self._calc_jaccard(valid_texts,
#                                        valid_input_ids,
#                                        valid_sentiments,
#                                        valid_selected_texts,
#                                        # valid_labels_head,
#                                        # valid_labels_tail,
#                                        valid_preds_head,
#                                        valid_preds_tail,
#                                        loader.dataset.tokenizer,
#                                        self.cfg_train['thresh_unit'],
#                                        self.cfg_predict['neutral_origin'],
#                                        self.cfg_predict['head_tail_equal_handle'],
#                                        )
#
#         valid_preds = (valid_preds_head, valid_preds_tail)
#         valid_labels = (valid_labels_head, valid_labels_tail)
#
#         return valid_loss, best_thresh, best_jaccard, valid_textIDs, \
#             valid_input_ids, valid_preds, valid_labels
#
#     def _get_predicted_texts(self, texts, input_ids, sentiments, y_preds_head,
#                              y_preds_tail, tokenizer,
#                              neutral_origin=False,
#                              head_tail_equal_handle='tail'):
#         predicted_texts = []
#         for text, input_id, sentiment, y_pred_head, y_pred_tail \
#                 in zip(texts, input_ids, sentiments, y_preds_head, y_preds_tail):
#             if neutral_origin and sentiment == 'neutral':
#                 predicted_texts.append(text)
#                 continue
#             pred_label_head = (
#                 y_pred_head - torch.cat([y_pred_head[-1:], y_pred_head[:-1]])).argmax()
#             pred_label_tail = (
#                 y_pred_tail - torch.cat([y_pred_tail[-1:], y_pred_tail[:-1]])).argmax()
#             if pred_label_head > pred_label_tail or len(text.split()) < 2:
#                 predicted_text = text
#             elif pred_label_head == pred_label_tail:
#                 if head_tail_equal_handle == 'nothing':
#                     predicted_text = ''
#                 elif head_tail_equal_handle == 'head':
#                     predicted_text = tokenizer.decode(
#                         input_id[pred_label_head:pred_label_tail + 1])
#                 elif head_tail_equal_handle == 'tail':
#                     predicted_text = tokenizer.decode(
#                         input_id[pred_label_head - 1:pred_label_tail])
#                 else:
#                     raise NotImplementedError()
#             else:
#                 predicted_text = tokenizer.decode(
#                     input_id[pred_label_head:pred_label_tail])
#             predicted_texts.append(predicted_text)
#
#         return predicted_texts
#
#     def _test_loop(self, model, loader, use_special_mask, use_offsets):
#         model.eval()
#         sigmoid = Sigmoid()
#
#         with torch.no_grad():
#             textIDs = []
#             test_texts = []
#             test_input_ids = []
#             test_offsets = []
#             test_sentiments = []
#             test_preds_head = []
#             test_preds_tail = []
#
#             for batch in tqdm(loader):
#                 textID = batch['textID']
#                 test_text = batch['text']
#                 input_ids = batch['input_ids'].to(self.device)
#                 if use_offsets:
#                     offsets = batch['offsets'].to(self.device)
#                 sentiment = batch['sentiment']
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                     if use_special_mask else None
#
#                 (logits, ) = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     special_tokens_mask=special_tokens_mask,
#                 )
#                 logits_head = logits[0]
#                 logits_tail = logits[1]
#
#                 predicted_head = sigmoid(logits_head.data)
#                 predicted_tail = sigmoid(logits_tail.data)
#
#                 test_texts.append(test_text)
#                 textIDs.append(textID)
#                 test_input_ids.append(input_ids.cpu())
#                 if use_offsets:
#                     test_offsets.append(offsets.cpu())
#                 test_sentiments.append(sentiment)
#                 test_preds_head.append(predicted_head.cpu())
#                 test_preds_tail.append(predicted_tail.cpu())
#
#             test_texts = list(chain.from_iterable(test_texts))
#             textIDs = list(chain.from_iterable(textIDs))
#             test_input_ids = torch.cat(test_input_ids)
#             if use_offsets:
#                 test_offsets = torch.cat(test_offsets)
#             test_sentiments = list(chain.from_iterable(test_sentiments))
#             test_preds_head = torch.cat(test_preds_head)
#             test_preds_tail = torch.cat(test_preds_tail)
#
#         return textIDs, test_texts, test_input_ids, test_offsets, \
#             test_sentiments, test_preds_head, test_preds_tail
#
#
# class r004HeadAnchorRunner(r002HeadTailRunner):
#
#     def _train_loop(self, model, optimizer, fobj,
#                     loader, warmup_batch, ema, accum_mod, use_specical_mask,
#                     fobj_segmentation, segmentation_loss_ratio):
#         model.train()
#         running_loss = 0
#
#         fobj_anchor = MSELoss()
#
#         for batch_i, batch in enumerate(tqdm(loader)):
#             if warmup_batch is not None:
#                 self._warmup(batch_i, warmup_batch, model)
#
#             input_ids = batch['input_ids'].to(self.device)
#             labels_head = batch['labels_head'].to(self.device)
#             labels_tail = batch['labels_tail'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
#             special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                 if use_specical_mask else None
#
#             (logits, ) = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 special_tokens_mask=special_tokens_mask,
#             )
#
#             logits_head = logits[0]
#             logits_tail = logits[1]
#
#             train_loss = fobj(logits_head, labels_head)
#             train_loss += fobj_anchor(logits_tail,
#                                       (labels_tail - labels_head).float())
#
#             if fobj_segmentation:
#                 labels_segmentation = batch['labels_segmentation']\
#                     .to(self.device)
#                 logits_segmentation = logits[2]
#                 # logits_segmentation *= attention_mask
#
#                 if self.cfg_fobj_segmentation['fobj_type'] == 'lovasz':
#                     train_loss += segmentation_loss_ratio * \
#                         fobj_segmentation(logits_segmentation,
#                                           labels_segmentation,
#                                           ignore=-1)
#                 else:
#                     # NOTE:
#                     # train_loss += segmentation_loss_ratio * \
#                     train_loss = segmentation_loss_ratio * \
#                         fobj_segmentation(logits_segmentation,
#                                           labels_segmentation)
#
#             train_loss.backward()
#
#             running_loss += train_loss.item()
#
#             if (batch_i + 1) % accum_mod == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#
#                 ema.on_batch_end(model)
#
#         train_loss = running_loss / len(loader)
#
#         return train_loss
#
#     def _valid_loop(self, model, fobj, loader, use_special_mask, use_offsets):
#         model.eval()
#         softmax = Softmax(dim=1)
#         running_loss = 0
#
#         fobj_anchor = MSELoss()
#
#         valid_textIDs_list = []
#         with torch.no_grad():
#             valid_texts = []
#             valid_textIDs = []
#             valid_input_ids = []
#             valid_offsets = []
#             valid_sentiments = []
#             valid_selected_texts = []
#             valid_preds_head, valid_preds_tail = [], []
#             valid_labels_head, valid_labels_tail = [], []
#             for batch in tqdm(loader):
#                 textIDs = batch['textID']
#                 valid_text = batch['text']
#                 input_ids = batch['input_ids'].to(self.device)
#                 if use_offsets:
#                     valid_offset = batch['offsets'].to(self.device)
#                 valid_sentiment = batch['sentiment']
#                 selected_texts = batch['selected_text']
#                 labels_head = batch['labels_head'].to(self.device)
#                 labels_tail = batch['labels_tail'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                     if use_special_mask else None
#
#                 (logits, ) = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     special_tokens_mask=special_tokens_mask,
#                 )
#                 logits_head = logits[0]
#                 logits_tail = logits[1]
#
#                 valid_loss = fobj(logits_head, labels_head)
#                 valid_loss += fobj_anchor(logits_tail,
#                                           labels_tail - labels_head)
#                 running_loss += valid_loss.item()
#
#                 # _, predicted = torch.max(outputs.data, 1)
#                 predicted_head = softmax(logits_head.data)
#                 predicted_tail = logits_tail.data
#
#                 valid_textIDs_list.append(textIDs)
#                 valid_texts.append(valid_text)
#                 valid_input_ids.append(input_ids.cpu())
#                 if use_offsets:
#                     valid_offsets.append(valid_offset.cpu())
#                 valid_sentiments.append(valid_sentiment)
#                 valid_selected_texts.append(selected_texts)
#                 valid_preds_head.append(predicted_head.cpu())
#                 valid_preds_tail.append(predicted_tail.cpu())
#                 valid_labels_head.append(labels_head.cpu())
#                 valid_labels_tail.append(labels_tail.cpu())
#
#             valid_loss = running_loss / len(loader)
#
#             valid_textIDs = list(
#                 itertools.chain.from_iterable(valid_textIDs_list))
#             valid_texts = list(itertools.chain.from_iterable(valid_texts))
#             valid_input_ids = torch.cat(valid_input_ids)
#             if use_offsets:
#                 valid_offsets = torch.cat(valid_offsets)
#             valid_sentiments = list(
#                 itertools.chain.from_iterable(valid_sentiments))
#             valid_selected_texts = list(
#                 itertools.chain.from_iterable(valid_selected_texts))
#             valid_preds_head = torch.cat(valid_preds_head)
#             valid_preds_tail = torch.cat(valid_preds_tail)
#             valid_labels_head = torch.cat(valid_labels_head)
#             valid_labels_tail = torch.cat(valid_labels_tail)
#
#             if use_offsets:
#                 best_thresh, best_jaccard = \
#                     self._calc_jaccard_offsets(valid_texts,
#                                                valid_offsets,
#                                                valid_sentiments,
#                                                valid_selected_texts,
#                                                valid_preds_head,
#                                                valid_preds_tail,
#                                                self.cfg_predict['neutral_origin'],
#                                                self.cfg_predict['head_tail_equal_handle'],
#                                                )
#             else:
#                 best_thresh, best_jaccard = \
#                     self._calc_jaccard(valid_texts,
#                                        valid_input_ids,
#                                        valid_sentiments,
#                                        valid_selected_texts,
#                                        # valid_labels_head,
#                                        # valid_labels_tail,
#                                        valid_preds_head,
#                                        valid_preds_tail,
#                                        loader.dataset.tokenizer,
#                                        self.cfg_train['thresh_unit'],
#                                        self.cfg_predict['neutral_origin'],
#                                        self.cfg_predict['head_tail_equal_handle'],
#                                        )
#
#         valid_preds = (valid_preds_head, valid_preds_tail)
#         valid_labels = (valid_labels_head, valid_labels_tail)
#
#         return valid_loss, best_thresh, best_jaccard, valid_textIDs, \
#             valid_input_ids, valid_preds, valid_labels
#
#     def _get_predicted_texts(self, texts, input_ids, sentiments, y_preds_head,
#                              y_preds_tail, tokenizer,
#                              neutral_origin=False,
#                              head_tail_equal_handle='tail'):
#         predicted_texts = []
#         for text, input_id, sentiment, y_pred_head, y_pred_tail \
#                 in zip(texts, input_ids, sentiments, y_preds_head, y_preds_tail):
#             if neutral_origin and sentiment == 'neutral':
#                 predicted_texts.append(text)
#                 continue
#             pred_label_head = y_pred_head.argmax()
#             pred_label_tail = (
#                 pred_label_head +
#                 y_pred_tail).round()[0].long()   # 四捨五入
#             if pred_label_head > pred_label_tail or len(text.split()) < 2 \
#                     or pred_label_tail >= len(input_ids):
#                 predicted_text = text
#             elif pred_label_head == pred_label_tail:
#                 if head_tail_equal_handle == 'nothing':
#                     predicted_text = ''
#                 elif head_tail_equal_handle == 'head':
#                     predicted_text = tokenizer.decode(
#                         input_id[pred_label_head:pred_label_tail + 1])
#                 elif head_tail_equal_handle == 'tail':
#                     predicted_text = tokenizer.decode(
#                         input_id[pred_label_head - 1:pred_label_tail])
#                 else:
#                     raise NotImplementedError()
#             else:
#                 predicted_text = tokenizer.decode(
#                     input_id[pred_label_head:pred_label_tail])
#             predicted_texts.append(predicted_text)
#
#         return predicted_texts
#
#
# class r005HeadTailRunner(r002HeadTailRunner):
#     def _train_loop(self, model, optimizer, fobj,
#                     loader, warmup_batch, ema, accum_mod, use_specical_mask,
#                     fobj_segmentation, segmentation_loss_ratio,
#                     loss_weight_type, fobj_index_diff):
#         model.train()
#         running_loss = 0
#
#         softargmax1d = SoftArgmax1D(
#             beta=5., device=self.device).to(
#             self.device)
#
#         for batch_i, batch in enumerate(tqdm(loader)):
#             if warmup_batch > 0:
#                 self._warmup(batch_i, warmup_batch, model)
#
#             input_ids = batch['input_ids'].to(self.device)
#             labels_head = batch['labels_head'].to(self.device)
#             labels_tail = batch['labels_tail'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
#             special_tokens_mask = batch['special_tokens_mask'].to(self.device) \
#                 if use_specical_mask else None
#
#             (logits, ) = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 special_tokens_mask=special_tokens_mask,
#             )
#
#             # 5 is temerature
#             logits_head = logits[0]
#             logits_tail = logits[1]
#
#             if loss_weight_type == 'sel_len':
#                 sel_len_weight = 1. * (
#                     1. / (labels_tail - labels_head).float())
#                 train_losses_head = fobj(logits_head, labels_head)
#                 train_loss = (train_losses_head * sel_len_weight).mean()
#                 train_losses_tail = fobj(logits_tail, labels_tail)
#                 train_loss += (train_losses_tail * sel_len_weight).mean()
#             elif loss_weight_type == 'sel_len_log':
#                 sel_len_weight = 1. * (
#                     1. / (labels_tail - labels_head).float() / 10. + 2.71828).log()
#                 # 1. / (labels_tail - labels_head).float() + 2.71828).log()
#                 train_losses_head = fobj(logits_head, labels_head)
#                 train_loss = (train_losses_head * sel_len_weight).mean()
#                 train_losses_tail = fobj(logits_tail, labels_tail)
#                 train_loss += (train_losses_tail * sel_len_weight).mean()
#             else:
#                 train_loss = self.cfg_train['head_ratio'] * \
#                     fobj(logits_head, labels_head)
#                 train_loss += self.cfg_train['tail_ratio'] * \
#                     fobj(logits_tail, labels_tail)
#
#             if fobj_segmentation:
#                 labels_segmentation_head = batch['labels_segmentation_head']\
#                     .to(self.device)
#                 labels_segmentation_tail = batch['labels_segmentation_tail']\
#                     .to(self.device)
#                 # labels_segmentation_head_rev = batch['labels_segmentation_head_rev']\
#                 #     .to(self.device)
#                 # labels_segmentation_tail_rev = batch['labels_segmentation_tail_rev']\
#                 #     .to(self.device)
#                 logits_segmentation_head = logits[2]
#                 logits_segmentation_tail = logits[3]
#                 # logits_segmentation_head_rev = logits[4]
#                 # logits_segmentation_tail_rev = logits[5]
#
#                 if self.cfg_fobj_segmentation['fobj_type'] == 'lovasz':
#                     # train_loss = segmentation_loss_ratio * \
#                     train_loss += segmentation_loss_ratio * \
#                         fobj_segmentation(logits_segmentation_head,
#                                           labels_segmentation_head,
#                                           ignore=-1)  # , weights=1./labels_segmentation_head.sum(dim=1))
#                     train_loss += segmentation_loss_ratio * \
#                         fobj_segmentation(logits_segmentation_tail,
#                                           labels_segmentation_tail,
#                                           ignore=-1)  # , weights=1./labels_segmentation_tail.sum(dim=1))
#                     # train_loss += 0.5 * segmentation_loss_ratio * \
#                     #     fobj_segmentation(logits_segmentation_head_rev,
#                     #                       labels_segmentation_head_rev,
#                     #                       ignore=-1)
#                     # train_loss += 0.5 * segmentation_loss_ratio * \
#                     #     fobj_segmentation(logits_segmentation_tail_rev,
#                     #                       labels_segmentation_tail_rev,
#                     #                       ignore=-1)
#                 else:
#                     raise NotImplementedError()
#
#             if fobj_index_diff:
#                 pred_index_head = softargmax1d(logits_head)
#                 pred_index_tail = softargmax1d(logits_tail)
#                 pred_index_diff = pred_index_tail - pred_index_head
#                 labels_index_diff = (labels_tail - labels_head).float()
#                 train_loss += 0.0003 * fobj_index_diff(pred_index_diff,
#                                                        labels_index_diff)
#                 # train_loss += 0.003 * fobj_index_diff(pred_index_head,
#                 #                                       labels_head.float())
#                 # train_loss += 0.003 * fobj_index_diff(pred_index_tail,
#                 #                                       labels_tail.float())
#
#             train_loss.backward()
#
#             running_loss += train_loss.item()
#
#             if (batch_i + 1) % accum_mod == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#
#                 ema.on_batch_end(model)
#
#         train_loss = running_loss / len(loader)
#
#         return train_loss
