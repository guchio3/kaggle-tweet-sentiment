import datetime
import itertools
import os
import random
import time
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from tools.datasets import (TSEHeadTailDataset, TSEHeadTailDatasetV2,
                            TSESegmentationDataset)
from tools.loggers import myLogger
from tools.metrics import jaccard
from tools.models import (BertModelWBinaryMultiLabelClassifierHead,
                          BertModelWDualMultiClassClassifierHead,
                          RobertaModelWDualMultiClassClassifierHead,
                          RobertaModelWDualMultiClassClassifierHeadV2)
from tools.schedulers import pass_scheduler
from tools.splitters import mySplitter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Sigmoid, Softmax
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

random.seed(71)
torch.manual_seed(71)


class Runner(object):
    def __init__(self, exp_id, checkpoint, device, debug, config):
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
        self.logger.info(f'exp_id: {exp_id}')
        self.logger.info(f'checkpoint: {checkpoint}')
        self.logger.info(f'debug: {debug}')
        self.logger.info(f'config: {config}')

        # unpack config info
        # uppercase means raaaw value
        self.cfg_SINGLE_FOLD = config['SINGLE_FOLD']
        # self.cfg_batch_size = config['batch_size']
        # self.cfg_max_epoch = config['max_epoch']
        self.cfg_split = config['split']
        self.cfg_loader = config['loader']
        self.cfg_dataset = config['dataset']
        self.cfg_fobj = config['fobj']
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
            checkpoint_epoch = checkpoint['current_epoch']
            self.histories = checkpoint['histories']
            iter_epochs = range(checkpoint_epoch,
                                self.cfg_train['max_epoch'], 1)
        else:
            iter_epochs = range(0, self.cfg_train['max_epoch'], 1)

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
                fold_trn_df = fold_trn_df.set_index('textID')
                for invalid_label_csv in self.cfg_invalid_labels:
                    invalid_label_df = pd.read_csv(invalid_label_csv)
                    for i, row in invalid_label_df.iterrows():
                        fold_trn_df.loc[row['textID'], 'selected_text'] = \
                            row['guchio_selected_text']
                fold_trn_df = fold_trn_df.reset_index()

            if 'rm_neutral' in self.cfg_train \
                    and self.cfg_train['rm_neutral']:
                fold_trn_df = fold_trn_df.query('sentiment != "neutral"')
            trn_loader = self._build_loader(mode='train', df=fold_trn_df,
                                            **self.cfg_loader)
            fold_val_df = trn_df.iloc[val_idx]
            val_loader = self._build_loader(mode='test', df=fold_val_df,
                                            **self.cfg_loader)

            # get fobj
            fobj = self._get_fobj(**self.cfg_fobj)

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

            epoch_start_time = time.time()
            epoch_best_jaccard = -1
            self.logger.info('start trainging !')
            for current_epoch in iter_epochs:
                if self.checkpoint and current_epoch <= checkpoint_epoch:
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
                trn_loss = self._train_loop(model, optimizer, fobj, trn_loader)
                val_loss, best_thresh, best_jaccard, val_textIDs, \
                    val_input_ids, val_preds, val_labels = \
                    self._valid_loop(model, fobj, val_loader)
                epoch_best_jaccard = max(epoch_best_jaccard, best_jaccard)

                self.logger.info(
                    f'epoch: {current_epoch} / '
                    + f'trn loss: {trn_loss:.5f} / '
                    + f'val loss: {val_loss:.5f} / '
                    + f'best val thresh: {best_thresh:.5f} / '
                    + f'best val jaccard: {best_jaccard:.5f} / '
                    + f'lr: {optimizer.param_groups[0]["lr"]:.5f} / '
                    + f'time: {int(time.time()-start_time)}sec')

                self.histories[fold_num]['trn_loss'].append(trn_loss)
                self.histories[fold_num]['val_loss'].append(val_loss)
                self.histories[fold_num]['val_jac'].append(best_jaccard)

                scheduler.step()

                # send to cpu
                model = model.to('cpu')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

                self._save_checkpoint(fold_num, current_epoch,
                                      model, optimizer, scheduler,
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
            line_message = f'{self.exp_id}: fini fold {fold_num} in {fold_time} min. \n' \
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
        line_message = f'{self.exp_id}: fini all trn. \n' \
            f'jaccard: {jac_mean}+-{jac_std}' \
            f'time: {trn_time}'

    def _get_fobj(self, fobj_type):
        if fobj_type == 'bce':
            fobj = BCEWithLogitsLoss()
        elif fobj_type == 'ce':
            fobj = CrossEntropyLoss()
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

    def _get_scheduler(self, scheduler_type, max_epoch, optimizer):
        if scheduler_type == 'pass':
            scheduler = pass_scheduler()
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(max_epoch * 0.8),
                    int(max_epoch * 0.9)
                ],
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            # scheduler examples:
            #     [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
            # if you want to use cosine annealing, use below scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epoch, eta_min=0.00002
            )
        else:
            raise Exception(f'invalid scheduler_type: {scheduler_type}')
        return scheduler

    def _build_loader(self, mode, df,
                      trn_sampler_type, trn_batch_size,
                      tst_sampler_type, tst_batch_size
                      ):
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

        if self.cfg_dataset['dataset_type'] == 'tse_segmentation_dataset':
            dataset = TSESegmentationDataset(mode=mode, df=df, logger=self.logger,
                                             debug=self.debug, **self.cfg_dataset)
        elif self.cfg_dataset['dataset_type'] == 'tse_headtail_dataset':
            dataset = TSEHeadTailDataset(mode=mode, df=df, logger=self.logger,
                                         debug=self.debug, **self.cfg_dataset)
        elif self.cfg_dataset['dataset_type'] == 'tse_headtail_dataset_v2':
            dataset = TSEHeadTailDatasetV2(mode=mode, df=df, logger=self.logger,
                                           debug=self.debug, **self.cfg_dataset)
        else:
            raise NotImplementedError()

        if sampler_type == 'sequential':
            sampler = SequentialSampler(data_source=dataset)
        elif sampler_type == 'random':
            sampler = RandomSampler(data_source=dataset)
        else:
            raise NotImplementedError(
                f'sampler_type: {sampler_type} is not '
                'implemented for mode: {mode}')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=os.cpu_count(),
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    def _warmup(self, current_epoch, warmup_epoch, model):
        if current_epoch == 0:
            module = model if self.device == 'cpu' else model.module
            for name, child in module.named_children():
                if 'classifier' in name:
                    self.logger.info(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    self.logger.info(name + ' is frozen')
                    for param in child.parameters():
                        param.requires_grad = False
        if current_epoch == warmup_epoch:
            self.logger.info("Turn on all the layers")
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    def _save_checkpoint(self, fold_num, current_epoch,
                         model, optimizer, scheduler,
                         val_textIDs, val_input_ids, val_preds, val_labels,
                         val_loss, best_thresh, best_jaccard):
        if not os.path.exists(f'./checkpoints/{self.exp_id}/{fold_num}'):
            os.makedirs(f'./checkpoints/{self.exp_id}/{fold_num}')
        # pth means pytorch
        cp_filename = f'./checkpoints/{self.exp_id}/{fold_num}/' \
            f'epoch_{current_epoch}_{val_loss:.5f}_{best_thresh:.5f}' \
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
            temp_metric = float(split_filename[4])
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


class r001SegmentationRunner(Runner):
    def __init__(self, exp_id, checkpoint, device, debug, config):
        super().__init__(exp_id, checkpoint, device, debug, config,
                         TSESegmentationDataset)

    # def predict(self):
    #     tst_ids = self._get_test_ids()
    #     if self.debug:
    #         tst_ids = tst_ids[:300]
    #     test_loader = self._build_loader(
    #         mode="test", ids=tst_ids, augment=None)
    #     best_loss, best_acc = self._load_best_model()
    #     test_ids, test_preds = self._test_loop(test_loader)

    #     submission_df = pd.read_csv(
    #         './mnt/inputs/origin/sample_submission.csv')
    #     submission_df = submission_df.set_index('id_code')
    #     submission_df.loc[test_ids, 'sirna'] = test_preds
    #     submission_df = submission_df.reset_index()
    #     filename_base = f'{self.exp_id}_{self.exp_time}_' \
    #         f'{best_loss:.5f}_{best_acc:.5f}'
    #     sub_filename = f'./mnt/submissions/{filename_base}_sub.csv'
    #     submission_df.to_csv(sub_filename, index=False)

    #     self.logger.info(f'Saved submission file to {sub_filename} !')
    #     line_message = f'Finished the whole pipeline ! \n' \
    #         f'Training time : {self.trn_time} min \n' \
    #         f'Best valid loss : {best_loss:.5f} \n' \
    #         f'Best valid acc : {best_acc:.5f}'
    #     self.logger.send_line_notification(line_message)

    def _train_loop(self, model, optimizer, fobj, loader):
        model.train()
        running_loss = 0

        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            (logits, ) = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

            train_loss = fobj(logits, labels)

            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            running_loss += train_loss.item()

        train_loss = running_loss / len(loader)

        return train_loss

    def _valid_loop(self, model, fobj, loader):
        model.eval()
        sigmoid = Sigmoid()
        running_loss = 0

        valid_textIDs_list = []
        with torch.no_grad():
            valid_textIDs, valid_input_ids, valid_preds, valid_labels \
                = [], [], [], []
            for batch in tqdm(loader):
                textIDs = batch['textID']
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                (logits, ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                valid_loss = fobj(logits, labels)
                running_loss += valid_loss.item()

                # _, predicted = torch.max(outputs.data, 1)
                predicted = sigmoid(logits.data)

                valid_textIDs_list.append(textIDs)
                valid_input_ids.append(input_ids.cpu())
                valid_preds.append(predicted.cpu())
                valid_labels.append(labels.cpu())

            valid_loss = running_loss / len(loader)

            valid_textIDs = list(
                itertools.chain.from_iterable(valid_textIDs_list))
            valid_input_ids = torch.cat(valid_input_ids)
            valid_preds = torch.cat(valid_preds)
            valid_labels = torch.cat(valid_labels)
            # valid_jac = self._calc_jac(
            #     valid_preds, valid_labels
            # )

            best_thresh, best_jaccard = \
                self._calc_jaccard(valid_input_ids,
                                   valid_labels.bool(),
                                   valid_preds,
                                   loader.dataset.tokenizer,
                                   self.cfg_train['thresh_unit'])

        return valid_loss, best_thresh, best_jaccard, valid_textIDs, \
            valid_input_ids, valid_preds, valid_labels

    # def _test_loop(self, loader):
    #     self.model.eval()

    #     test_ids = []
    #     test_preds = []

    #     sel_log('predicting ...', self.logger)
    #     AUGNUM = 2
    #     with torch.no_grad():
    #         for (ids, images, labels) in tqdm(loader):
    #             images, labels = images.to(
    #                 self.device, dtype=torch.float), labels.to(
    #                 self.device)
    #             outputs = self.model.forward(images)
    #             # avg predictions
    #             # outputs = torch.mean(outputs.reshape((-1, 1108, 2)), 2)
    #             # outputs = torch.mean(torch.stack(
    #             #     [outputs[i::AUGNUM] for i in range(AUGNUM)], dim=2), dim=2)
    #             # _, predicted = torch.max(outputs.data, 1)
    #             sm_outputs = softmax(outputs, dim=1)
    #             sm_outputs = torch.mean(torch.stack(
    #                 [sm_outputs[i::AUGNUM] for i in range(AUGNUM)], dim=2), dim=2)
    #             _, predicted = torch.max(sm_outputs.data, 1)

    #             test_ids.append(ids[::2])
    #             test_preds.append(predicted.cpu())

    #         test_ids = np.concatenate(test_ids)
    #         test_preds = torch.cat(test_preds).numpy()

    #     return test_ids, test_preds

    def _calc_jaccard(self, input_ids, selected_text_masks,
                      y_preds, tokenizer, thresh_unit):
        best_thresh = -1
        best_jaccard = -1

        self.logger.info('now calcurating the best threshold for jaccard ...')
        for thresh in tqdm(list(np.arange(0.1, 1.0, thresh_unit))):
            # get predicted texts
            predicted_text_masks = [y_pred > thresh for y_pred in y_preds]
            # calc jaccard for this threshold
            temp_jaccard = 0
            for input_id, selected_text_mask, predicted_text_mask in zip(
                    input_ids, selected_text_masks, predicted_text_masks):
                selected_text = tokenizer.decode(
                    input_id[selected_text_mask])
                # fill continuous zeros between one
                _non_zeros = predicted_text_mask.nonzero()
                if _non_zeros.shape[0] > 0:
                    _predicted_text_mask_min = _non_zeros.min()
                    _predicted_text_mask_max = _non_zeros.max()
                    predicted_text_mask[_predicted_text_mask_min:
                                        _predicted_text_mask_max + 1] = True
                predicted_text = tokenizer.decode(
                    input_id[predicted_text_mask])
                temp_jaccard += jaccard(selected_text, predicted_text)

            temp_jaccard /= len(selected_text_masks)
            # update the best jaccard
            if temp_jaccard > best_jaccard:
                best_thresh = thresh
                best_jaccard = temp_jaccard

        assert best_thresh != -1
        assert best_jaccard != -1

        return best_thresh, best_jaccard


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

            textIDs, test_texts, test_input_ids, \
                test_sentiments, test_preds_head, test_preds_tail\
                = self._test_loop(model, tst_loader)

            fold_test_preds_heads.append(test_preds_head)
            fold_test_preds_tails.append(test_preds_tail)

        avg_test_preds_head = torch.mean(
            torch.stack(fold_test_preds_heads), dim=0)
        avg_test_preds_tail = torch.mean(
            torch.stack(fold_test_preds_tails), dim=0)
        predicted_texts = self._get_predicted_texts(
            test_texts,
            test_input_ids,
            test_sentiments,
            avg_test_preds_head,
            avg_test_preds_tail,
            tst_loader.dataset.tokenizer,
        )

        return textIDs, predicted_texts

    def _train_loop(self, model, optimizer, fobj, loader):
        model.train()
        running_loss = 0

        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            labels_head = batch['labels_head'].to(self.device)
            labels_tail = batch['labels_tail'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            (logits, ) = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits_head, logits_tail = logits

            train_loss = fobj(logits_head, labels_head)
            train_loss += fobj(logits_tail, labels_tail)

            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            running_loss += train_loss.item()

        train_loss = running_loss / len(loader)

        return train_loss

    def _valid_loop(self, model, fobj, loader):
        model.eval()
        softmax = Softmax()
        running_loss = 0

        valid_textIDs_list = []
        with torch.no_grad():
            valid_texts, valid_textIDs, valid_input_ids, valid_sentiments, valid_selected_texts = [], [], [], [], []
            valid_preds_head, valid_preds_tail = [], []
            valid_labels_head, valid_labels_tail = [], []
            for batch in tqdm(loader):
                textIDs = batch['textID']
                valid_text = batch['text']
                input_ids = batch['input_ids'].to(self.device)
                valid_sentiment = batch['sentiment']
                selected_texts = batch['selected_text']
                labels_head = batch['labels_head'].to(self.device)
                labels_tail = batch['labels_tail'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                (logits, ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_head, logits_tail = logits

                valid_loss = fobj(logits_head, labels_head)
                valid_loss += fobj(logits_tail, labels_tail)
                running_loss += valid_loss.item()

                # _, predicted = torch.max(outputs.data, 1)
                predicted_head = softmax(logits_head.data)
                predicted_tail = softmax(logits_tail.data)

                valid_textIDs_list.append(textIDs)
                valid_texts.append(valid_text)
                valid_input_ids.append(input_ids.cpu())
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
            valid_sentiments = list(
                itertools.chain.from_iterable(valid_sentiments))
            valid_selected_texts = list(
                itertools.chain.from_iterable(valid_selected_texts))
            valid_preds_head = torch.cat(valid_preds_head)
            valid_preds_tail = torch.cat(valid_preds_tail)
            valid_labels_head = torch.cat(valid_labels_head)
            valid_labels_tail = torch.cat(valid_labels_tail)

            best_thresh, best_jaccard = \
                self._calc_jaccard(valid_texts,
                                   valid_input_ids,
                                   valid_sentiments,
                                   valid_selected_texts,
                                   # valid_labels_head,
                                   # valid_labels_tail,
                                   valid_preds_head,
                                   valid_preds_tail,
                                   loader.dataset.tokenizer,
                                   self.cfg_train['thresh_unit'])

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

    def _calc_jaccard(self, texts, input_ids, sentiments, selected_texts,
                      y_preds_head, y_preds_tail, tokenizer, thresh_unit):

        temp_jaccard = 0
        for text, input_id, sentiment, selected_text, y_pred_head, y_pred_tail \
                in zip(texts, input_ids, sentiments, selected_texts,
                       y_preds_head, y_preds_tail):

            if self.cfg_predict['neutral_origin'] and sentiment == 'neutral':
                predicted_text = text
            else:
                pred_label_head = y_pred_head.argmax()
                pred_label_tail = y_pred_tail.argmax()
                if pred_label_head > pred_label_tail:
                    predicted_text = text
                else:
                    predicted_text = tokenizer.decode(
                        input_id[pred_label_head:pred_label_tail])
            temp_jaccard += jaccard(selected_text, predicted_text)

        best_thresh = -1
        best_jaccard = temp_jaccard / len(input_ids)

        return best_thresh, best_jaccard

    def _test_loop(self, model, loader):
        model.eval()
        softmax = Softmax()

        with torch.no_grad():
            textIDs, test_texts, test_input_ids, test_sentiments, test_preds_head, test_preds_tail = [], [], [], [], [], []
            for batch in tqdm(loader):
                textID = batch['textID']
                test_text = batch['text']
                input_ids = batch['input_ids'].to(self.device)
                sentiment = batch['sentiment']
                attention_mask = batch['attention_mask'].to(self.device)

                (logits, ) = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_head, logits_tail = logits

                predicted_head = softmax(logits_head.data)
                predicted_tail = softmax(logits_tail.data)

                test_texts.append(test_text)
                textIDs.append(textID)
                test_input_ids.append(input_ids.cpu())
                test_sentiments.append(sentiment)
                test_preds_head.append(predicted_head.cpu())
                test_preds_tail.append(predicted_tail.cpu())

            test_texts = list(chain.from_iterable(test_texts))
            textIDs = list(chain.from_iterable(textIDs))
            test_input_ids = torch.cat(test_input_ids)
            test_sentiments = list(chain.from_iterable(test_sentiments))
            test_preds_head = torch.cat(test_preds_head)
            test_preds_tail = torch.cat(test_preds_tail)

        return textIDs, test_texts, test_input_ids, \
            test_sentiments, test_preds_head, test_preds_tail

    def _get_predicted_texts(self, texts, input_ids, sentiments, y_preds_head,
                             y_preds_tail, tokenizer):
        predicted_texts = []
        for text, input_id, sentiment, y_pred_head, y_pred_tail \
                in zip(texts, input_ids, sentiments, y_preds_head, y_preds_tail):
            if self.cfg_predict['neutral_origin'] and sentiment == 'neutral':
                predicted_texts.append(text)
                continue
            pred_label_head = y_pred_head.argmax()
            pred_label_tail = y_pred_tail.argmax()
            if pred_label_head > pred_label_tail:
                predicted_text = text
            else:
                predicted_text = tokenizer.decode(
                    input_id[pred_label_head:pred_label_tail])
            predicted_texts.append(predicted_text)

        return predicted_texts
