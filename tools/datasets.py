from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


class TSEDataset(Dataset):
    def __init__(self, mode, tokenizer_type, pretrained_model_name_or_path,
                 do_lower_case, max_length, df, logger=None, debug=False):
        self.mode = mode
        if tokenizer_type == 'bert':
            self.tokenizer = BertTokenizer\
                .from_pretrained(
                    pretrained_model_name_or_path,
                    do_lower_case=do_lower_case)
        else:
            err_msg = f'{tokenizer_type} is not ' \
                'implemented for TSEDataset.'
            raise NotImplementedError(err_msg)
        self.max_length = max_length
        self.df = df.reset_index(drop=True)
        for i, row in self.df.iterrows():
            self.df.loc[i, 'text'] = f'{row["sentiment"]} ' + str(row['text'])
        self.df['input_ids'] = None
        self.df['labels'] = None
        self.df['attention_mask'] = None
        self.logger = logger
        self.debug = debug

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def _prep_text(self, row):
        raise NotImplementedError()


class TSESegmentationDataset(TSEDataset):
    def __getitem__(self, idx):
        row = self.df.loc[idx]

        if row['input_ids'] is None:
            row = self._prep_text(row)
            self.df.loc[idx, 'input_ids'] = row['input_ids']
            self.df.loc[idx, 'labels'] = row['labels']
            self.df.loc[idx, 'attention_mask'] = row['attention_mask']

        return {
            'textID': row['textID'],
            'input_ids': torch.tensor(row['input_ids']),
            'labels': torch.tensor(row['labels']),
            'attention_mask': torch.tensor(row['attention_mask']),
        }

    def _prep_text(self, row):
        text_output = self.tokenizer.encode_plus(
            text=row['text'],
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensor='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        selected_text_output = self.tokenizer.encode_plus(
            text=row['selected_text'],
            text_pair=None,
            add_special_tokens=False,
            max_length=self.max_length,
            pad_to_max_length=False,
            return_tensor='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # allign labels for segmentation
        input_ids = text_output['input_ids']
        sel_input_ids = selected_text_output['input_ids']
        matched_cnt = len([i for i in input_ids[:len(sel_input_ids)]
                           if i in sel_input_ids])
        best_matched_cnt = matched_cnt
        best_matched_i = 0
        # for i in range(0, len(input_ids)):
        for i in range(0, len(input_ids) - len(sel_input_ids)):
            head_input_id_i = input_ids[i]
            tail_input_id_i = input_ids[i + len(sel_input_ids)]
            if head_input_id_i in sel_input_ids:
                matched_cnt -= 1
            if tail_input_id_i in sel_input_ids:
                matched_cnt += 1
            if matched_cnt < 0:
                raise Exception('invalid logic')

            if best_matched_cnt < matched_cnt:
                best_matched_cnt = matched_cnt
                best_matched_i = i + 1   # 抜いた時の話なので
            if best_matched_cnt == len(sel_input_ids):
                break
            # if input_id_i == sel_input_ids[0]:
            # if input_id_i in sel_input_ids:
            #     temp_matched_len = 0
            #     for t_1, t_2 in zip(
            #             input_ids[i:i + len(sel_input_ids)], sel_input_ids):
            #         if t_1 == t_2:
            #             temp_matched_len += 1
            #     if best_matched_len < temp_matched_len:
            #         best_matched_len = temp_matched_len
            #         best_matched_i = i
            #     if best_matched_len == len(sel_input_ids):
            #         break
        if best_matched_cnt == 0:
            print('===============================')
            print(row)
            print('===============================')
            print(input_ids)
            print(sel_input_ids)

        labels = np.zeros(len(input_ids))
        labels[list(range(best_matched_i,
                          best_matched_i + len(sel_input_ids)))] = 1

        row['input_ids'] = text_output['input_ids']
        row['labels'] = labels
        row['attention_mask'] = text_output['attention_mask']
        return row


class TSEHeadTailDataset(TSEDataset):
    def __getitem__(self, idx):
        row = self.df.loc[idx]

        if row['input_ids'] is None:
            row = self._prep_text(row)
            self.df.loc[idx, 'input_ids'] = row['input_ids']
            self.df.loc[idx, 'labels'] = row['labels']
            self.df.loc[idx, 'attention_mask'] = row['attention_mask']

        return {
            'textID': row['textID'],
            'input_ids': torch.tensor(row['input_ids']),
            'attention_mask': torch.tensor(row['attention_mask']),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
        }

    def _prep_text(self, row):
        text_output = self.tokenizer.encode_plus(
            text=row['text'],
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensor='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        row['input_ids'] = text_output['input_ids']
        row['attention_mask'] = text_output['attention_mask']
        if 'selected_text' not in row:
            row['labels_head'] = -1
            row['labels_tail'] = -1
            return row

        selected_text_output = self.tokenizer.encode_plus(
            text=row['selected_text'],
            text_pair=None,
            add_special_tokens=False,
            max_length=self.max_length,
            pad_to_max_length=False,
            return_tensor='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # allign labels for segmentation
        input_ids = text_output['input_ids']
        sel_input_ids = selected_text_output['input_ids']
        matched_cnt = len([i for i in input_ids[:len(sel_input_ids)]
                           if i in sel_input_ids])
        best_matched_cnt = matched_cnt
        best_matched_i = 0
        # for i in range(0, len(input_ids)):
        for i in range(0, len(input_ids) - len(sel_input_ids)):
            head_input_id_i = input_ids[i]
            tail_input_id_i = input_ids[i + len(sel_input_ids)]
            if head_input_id_i in sel_input_ids:
                matched_cnt -= 1
            if tail_input_id_i in sel_input_ids:
                matched_cnt += 1
            if matched_cnt < 0:
                raise Exception('invalid logic')

            if best_matched_cnt < matched_cnt:
                best_matched_cnt = matched_cnt
                best_matched_i = i + 1   # 抜いた時の話なので
            if best_matched_cnt == len(sel_input_ids):
                break

        if best_matched_cnt == 0:
            print('===============================')
            print(row)
            print('===============================')
            print(input_ids)
            print(sel_input_ids)
            print(selected_text_output['input_ids'])

        row['labels_head'] = best_matched_i
        row['labels_tail'] = best_matched_i + len(sel_input_ids)
        return row
