import datetime
import itertools
import os
import random
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm

from tools.datasets import TSEDataset
from tools.loggers import myLogger
from tools.metrics import jaccard
from tools.models import BertModelWBinaryMultiLabelClassifierHead
from tools.schedulers import pass_scheduler
from tools.splitters import mySplitter

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
        self.logger = myLogger(f'./logs/{self.exp_id}_{self.exp_time}.log')
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
        self.cfg_invalid_labels = config['invalid_labels'] \
            if 'invalid_labels' in config else None

        self.histories = {
            'train_loss': [],
            'valid_loss': [],
            'valid_acc': [],
        }

    def train(self):
        # load and preprocess train.csv
        trn_df = pd.read_csv('./inputs/origin/train.csv')
        trn_df = trn_df[trn_df.text.notnull()].reset_index(drop=True)
        if self.cfg_invalid_labels:
            trn_df = trn_df.set_index('textID')
            for invalid_label_csv in self.cfg_invalid_labels:
                invalid_label_df = pd.read_csv(invalid_label_csv)
                for i, row in invalid_label_df.iterrows():
                    trn_df.loc[row['textID'], 'selected_text'] = \
                        row['guchio_selected_text']
            trn_df = trn_df.reset_index()

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
            optimizer = self._get_optimizer(model=model, **self.cfg_optimizer)
            scheduler = self._get_scheduler(optimizer=optimizer,
                                            max_epoch=self.cfg_train['max_epoch'],
                                            **self.cfg_scheduler)
            if self.checkpoint and checkpoint_fold_num == fold_num:
                model.module.load_state_dict(checkpoint['model_state_dict'])
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

                self._warmup(current_epoch, self.cfg_train['warmup_epoch'],
                             model)
                trn_loss = self._train_loop(model, optimizer, fobj, trn_loader)
                val_loss, val_textIDs, best_thresh, best_jaccard, \
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
                # self.histories[fold_num]['val_jac'].append(val_jac)

                scheduler.step()

                # send to cpu
                model = model.to('cpu')

                self._save_checkpoint(fold_num, current_epoch,
                                      model, optimizer, scheduler,
                                      val_textIDs, val_input_ids, val_preds,
                                      val_labels, val_loss,
                                      best_thresh, best_jaccard)

            fold_time = int(time.time() - epoch_start_time) // 60
            line_message = f'fini fold {fold_num} in {fold_time} min. \n' \
                f'epoch best jaccard: {epoch_best_jaccard}'
            self.logger.send_line_notification(line_message)

            if self.cfg_SINGLE_FOLD:
                break

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

    def _get_fobj(self, fobj_type):
        if fobj_type == 'bce':
            fobj = BCEWithLogitsLoss()
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, num_labels,
                   pretrained_model_name_or_path):
        if model_type == 'bert-segmentation':
            model = BertModelWBinaryMultiLabelClassifierHead(
                num_labels,
                pretrained_model_name_or_path
            )
        else:
            raise Exception(f'invalid model_type: {model_type}')
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
                optimizer, T_max=max_epoch, eta_min=0.00001
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

        dataset = TSEDataset(mode=mode, df=df, logger=self.logger,
                             debug=self.debug, **self.cfg_dataset)
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
            for name, child in model.module.named_children():
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
                    labels=labels,
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
                self._calc_best_threshold_for_jaccard(valid_labels,
                                                      valid_preds,
                                                      loader.dataset.tokenizer)

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

    def _get_predicted_text(self, y_pred, thresh, tokenizer):
        selected_ids = y_pred[y_pred > thresh]
        predicted_text = tokenizer.decode(selected_ids)
        return predicted_text

    def _calc_best_threshold_for_jaccard(self, selected_texts,
                                         y_preds, tokenizer):
        best_thresh = -1
        best_jaccard = -1

        self.logger.info('now calcurating the best threshold for jaccard ...')
        for thresh in np.arange(0.1, 1.0, 0.01):
            # get predicted texts
            predicted_texts = [
                self._get_predicted_text(y_pred, thresh, tokenizer)
                for y_pred in y_preds]
            # calc jaccard for this threshold
            temp_jaccard = 0
            for selected_text, predicted_text in zip(
                    selected_texts, predicted_texts):
                temp_jaccard += jaccard(selected_text, predicted_text)
            temp_jaccard /= len(selected_texts)
            # update the best jaccard
            if temp_jaccard > best_jaccard:
                best_thresh = thresh
                best_jaccard = temp_jaccard

        assert best_thresh != -1
        assert best_jaccard != -1
        return best_thresh, best_jaccard

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
        cp_dict = {
            'fold_num': fold_num,
            'current_epoch': current_epoch,
            'model_state_dict': model.module.state_dict(),
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
        best_loss = np.inf
        # best_metric = -1
        best_filename = ''
        for filename in glob(f'./checkpoints/{self.exp_id}/{fold_num}/*'):
            split_filename = filename.split('/')[-1].split('_')
            temp_loss = float(split_filename[2])
            # temp_metric = float(split_filename[3])
            # if temp_metric > best_metric:
            if temp_loss < best_loss:
                best_filename = filename
                best_loss = temp_loss
                # best_metric = temp_metric
        return best_filename  # , best_loss, best_acc

    def _load_best_checkpoint(self, fold_num):
        best_cp_filename = self._search_best_filename(fold_num)
        self.logger.info(f'the best file is {best_cp_filename} !')
        best_checkpoint = torch.load(best_cp_filename)
        return best_checkpoint
