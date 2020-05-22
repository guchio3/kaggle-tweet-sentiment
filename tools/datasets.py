from abc import abstractmethod

import numpy as np

import torch
from tools.tokenizers import (myBertByteLevelBPETokenizer,
                              myRobertaByteLevelBPETokenizer)
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer


class TSEDataset(Dataset):
    def __init__(self, mode, tokenizer_type, pretrained_model_name_or_path,
                 do_lower_case, max_length, df,
                 logger=None, debug=False, add_pair_prefix_space=True):
        self.mode = mode
        self.add_pair_prefix_space = add_pair_prefix_space
        if tokenizer_type == 'bert':
            self.tokenizer = BertTokenizer\
                .from_pretrained(
                    pretrained_model_name_or_path,
                    do_lower_case=do_lower_case)
        elif tokenizer_type == 'roberta':
            self.tokenizer = RobertaTokenizer\
                .from_pretrained(
                    pretrained_model_name_or_path,
                    do_lower_case=do_lower_case)
        elif tokenizer_type == 'bert_bytelevel_bpe':
            self.tokenizer = myBertByteLevelBPETokenizer(
                vocab_file=f'{pretrained_model_name_or_path}/bert/tokenizer/vocab.json',
                # merges_file=f'{pretrained_model_name_or_path}/bert/tokenizer/merges.txt',
                lowercase=do_lower_case,
                add_prefix_space=add_pair_prefix_space
            )
        elif tokenizer_type == 'roberta_bytelevel_bpe':
            self.tokenizer = myRobertaByteLevelBPETokenizer(
                vocab_file=f'{pretrained_model_name_or_path}/roberta/tokenizer/vocab.json',
                merges_file=f'{pretrained_model_name_or_path}/roberta/tokenizer/merges.txt',
                lowercase=do_lower_case,
                add_prefix_space=add_pair_prefix_space
            )
        else:
            err_msg = f'{tokenizer_type} is not ' \
                'implemented for TSEDataset.'
            raise NotImplementedError(err_msg)
        self.max_length = max_length
        self.df = df.reset_index(drop=True)
        # for i, row in self.df.iterrows():
        #     self.df.loc[i, 'text'] = f'[{row["sentiment"]}] ' \
        #         + str(row['text'])
        #     # self.df.loc[i, 'text'] = str(row['text']) + f' [SEP] [{row["sentiment"]}]'
        # self.tokenizer.add_tokens([
        #     '[neutral]',
        #     '[positive]',
        #     '[negative]',
        # ])
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

        # if row['input_ids'] is None:
        row = self._prep_text(row)
        #     # for key in row.to_dict().keys():
        #     #     self.df.loc[idx, key] = row[key]
        #     # self.df.loc[idx, 'input_ids'] = row['input_ids']
        #     # self.df.loc[idx, 'labels'] = row['labels']
        #     # self.df.loc[idx, 'attention_mask'] = row['attention_mask']

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']),
            'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
        }

    def _prep_text(self, row):
        text = " " + " ".join(row['text'].split())
        text_output = self.tokenizer.encode_plus(
            text=text,
            # text_pair=None,
            # text_pair=f"[{row['sentiment']}]",
            text_pair=row['sentiment'],
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensor='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        row['input_ids'] = text_output['input_ids']
        row['attention_mask'] = text_output['attention_mask']
        row['special_tokens_mask'] = text_output['special_tokens_mask']
        if 'selected_text' not in row:
            row['selected_text'] = ''
            row['labels_head'] = -1
            row['labels_tail'] = -1
            return row

        text = " " + " ".join(row['selected_text'].split())
        selected_text_output = self.tokenizer.encode_plus(
            text=text,
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
        # 1 start なのは、先頭の token をスルーするため
        matched_cnt = len([i for i in input_ids[1:1 + len(sel_input_ids)]
                           if i in sel_input_ids])
        best_matched_cnt = matched_cnt
        best_matched_i = 1
        # for i in range(0, len(input_ids)):
        # 1 start なのは、先頭の token をスルーするため
        for i in range(1, len(input_ids) - len(sel_input_ids)):
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
            self.logger.debug('===============================')
            self.logger.debug(row)
            self.logger.debug('===============================')
            self.logger.debug(input_ids)
            self.logger.debug(sel_input_ids)
            self.logger.debug(selected_text_output['input_ids'])
            # self.logger.debug(f'textID: {row["textID"]} -- no matching.')

        row['labels_head'] = best_matched_i
        row['labels_tail'] = best_matched_i + len(sel_input_ids)
        # 時々ラベルミスで sel_input_ids の方が長くなる
        # -1 for sentiment
        # i_length = min(
        #     len(sel_input_ids),
        #     (row['special_tokens_mask'] == 0).sum() - 1)
        # row['labels_tail'] = best_matched_i + i_length
        return row


class TSEHeadTailDatasetV2(TSEDataset):
    '''
    use kernal text preprocess logic

    '''

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        if row['input_ids'] is None:
            row = self._prep_text(row)
            self.df.loc[idx, 'input_ids'] = row['input_ids']
            self.df.loc[idx, 'labels'] = row['labels']
            self.df.loc[idx, 'attention_mask'] = row['attention_mask']

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']).long(),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']).long(),
            # 'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
        }

    def _prep_text(self, row):
        if self.add_pair_prefix_space:
            sentiment_id = {
                'positive': [1313],
                'negative': [2430],
                'neutral': [7974]}
        else:
            sentiment_id = {
                'positive': [22173],
                'negative': [33407],
                'neutral': [12516]}
        input_ids = np.ones(self.max_length, dtype='int32')
        attention_mask = np.zeros(self.max_length, dtype='int32')
        # token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
        # start_tokens = np.zeros(self.max_length, dtype='int32')
        # end_tokens = np.zeros(self.max_length, dtype='int32')

        # this is test case
        if 'selected_text' not in row:
            row['selected_text'] = ''

        # FIND OVERLAP
        text1 = " " + " ".join(row['text'].split())
        text2 = " " + " ".join(row['selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = self.tokenizer.encode(text1)

        # ID_OFFSETS
        offsets = []
        idx = 0
        for t in enc.ids:
            w = self.tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # START END TOKENS
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        # s_tok = self.tokenizer.encode(f'[{row["sentiment"]}]').ids
        s_tok = sentiment_id[row['sentiment']]
        input_ids[:len(enc.ids) + 5] = [0] + \
            enc.ids + [2, 2] + s_tok + [2]
        attention_mask[:len(enc.ids) + 5] = 1
        # if len(toks) > 0:
        #     start_tokens[toks[0] + 1] = 1
        #     end_tokens[toks[-1] + 1] = 1

        row['input_ids'] = input_ids
        row['attention_mask'] = attention_mask
        if len(toks) > 0:
            row['labels_head'] = toks[0] + 1  # +1 は [0] を除去するため
            row['labels_tail'] = toks[-1] + 1 + 1  # +1+1 は toks[-1] も使うため
        else:
            row['labels_head'] = 1
            row['labels_tail'] = 1 + len(enc.ids)  # == len(offsets)

        return row


class TSEHeadTailDatasetV3(TSEDataset):
    '''
    use kernal text preprocess logic
    https://www.kaggle.com/abhishek/roberta-inference-5-folds

    '''

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        if row['input_ids'] is None:
            row = self._prep_text(row)
            self.df.loc[idx, 'input_ids'] = row['input_ids']
            self.df.loc[idx, 'labels'] = row['labels']
            self.df.loc[idx, 'attention_mask'] = row['attention_mask']

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']).long(),
            'offsets': torch.tensor(row['offsets']).long(),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']).long(),
            # 'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
        }

    def _prep_text(self, row):
        if self.add_pair_prefix_space:
            sentiment_id = {
                'positive': [1313],
                'negative': [2430],
                'neutral': [7974]}
        else:
            sentiment_id = {
                'positive': [22173],
                'negative': [33407],
                'neutral': [12516]}

        # this is test case
        if 'selected_text' not in row:
            row['selected_text'] = ''

        tweet = " " + " ".join(row['text'].split())
        selected_text = " " + " ".join(row['selected_text'].split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        input_ids = [0] + sentiment_id[row['sentiment']] + \
            [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        row['input_ids'] = input_ids
        row['attention_mask'] = mask
        row['token_type_ids'] = token_type_ids
        row['labels_head'] = targets_start
        row['labels_tail'] = targets_end + 1
        row['offsets'] = tweet_offsets

        return row


class TSEHeadTailSegmentationDataset(TSEHeadTailDataset):

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        row = self._prep_text(row)

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']),
            'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
            'labels_segmentation': torch.tensor(row['labels_segmentation']),
        }

    def _prep_text(self, row):
        row = super()._prep_text(row)
        labels_segmentation = np.zeros(self.max_length)
        if row['labels_head'] >= 0 and row['labels_tail'] >= 0:
            labels_segmentation[row['labels_head']:row['labels_tail']] = 1
        row['labels_segmentation'] = labels_segmentation
        pad_token = self.tokenizer.encode_plus(
            self.tokenizer.special_tokens_map['pad_token'],
            text_pair=None,
            add_special_tokens=False,
            max_length=1)['input_ids'][0]
        row['labels_segmentation'][np.asarray(
            row['input_ids']) != pad_token] = -1
        return row
