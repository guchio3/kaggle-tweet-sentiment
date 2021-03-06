import re
from abc import abstractmethod
import pandas as pd

import numpy as np

import torch
from tools.tokenizers import (myBertByteLevelBPETokenizer,
                              myRobertaByteLevelBPETokenizer)
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer


def remove_http(text, selected_text):
    len_st = len(selected_text) - 1
    http_len = len("<HTTP>")
    http_offset = 0
    org_text = text
    new_selected_text = selected_text

    for ind in (i for i, e in enumerate(text) if e == selected_text[1]):
        if " " + text[ind: ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1  # 最後のindex

    find_idx = 0
#     print(selected_text)
    for i, vocab in enumerate(text.split(" ")):
        if "http" in vocab:
            http_idx0 = org_text.find(vocab, find_idx)
            len_vocab = len(vocab)
            http_idx1 = http_idx0 + len_vocab - 1  # 最後のindex
            http_vocab = vocab
            text = text.replace(http_vocab, "<HTTP>")
            find_idx = http_idx1  # 複数回出現する場合に検索範囲を更新する
#             print(f" vocab:{http_vocab}\n org_text:{org_text} h0:{http_idx0},h1:{http_idx1}")

            if http_idx0 <= idx1 and http_idx1 >= idx0:
                dup_idx0 = max(idx0, http_idx0)
                dup_idx1 = min(idx1, http_idx1)
                dup_idx0_s = dup_idx0 - (idx0 - 1)  # in org selected
                dup_idx1_s = dup_idx1 - (idx0 - 1)  # in org selected

                new_selected_text = new_selected_text[:http_offset + dup_idx0_s] + \
                    "<HTTP>" + new_selected_text[http_offset + dup_idx1_s + 1:]
                http_offset += http_len - (dup_idx1_s - dup_idx0_s + 1)

    return text, new_selected_text


class TSEDataset(Dataset):
    def __init__(self, mode, tokenizer_type, pretrained_model_name_or_path,
                 do_lower_case, max_length, df,
                 logger=None, debug=False, add_pair_prefix_space=True,
                 tokenize_period=False, tail_index='natural',
                 use_magic=False, tkm_annot=False):
        self.mode = mode
        self.add_pair_prefix_space = add_pair_prefix_space
        self.tokenize_period = tokenize_period
        self.tail_index = tail_index
        self.use_magic = use_magic
        if tkm_annot and mode == 'train':
            df_tkm_st = pd.read_csv('./inputs/datasets/tkm_annot/tkm_annotated.csv')
            map_tkm_st = df_tkm_st.to_dict()['tkm_selected_text']
            map_tkm_st['22f06df70a'] = 'i wish'
            map_tkm_st['0565804d90'] = ' one day my hugs will come    *fingers still crossed*'
            self.map_tkm_st = map_tkm_st
        else:
            self.map_tkm_st = {}
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
        if tokenize_period:
            added_num = self.tokenizer.add_tokens([
                '[S]',
                '[PERIOD]',
                '[EXCL]',
                '[QUES]',
            ])
            logger.info(f'added {added_num} tokens.')
            example = self.tokenizer.encode("[S][PERIOD][EXCL][QUES]")
            logger.info(f'ex. "[S][PERIOD][EXCL][QUES]" -> {example.ids}')

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
            row['selected_text'] = row['text']

        if self.use_magic:
            selected_text = row['selected_text']
            ss = row['text'].find(selected_text)
            # selected text の前に space が 1 or 2 個あったらそれに合わせる
            if row['text'][max(ss - 2, 0):ss] == '  ':
                ss -= 2
            if ss > 0 and row['text'][ss - 1] == ' ':
                ss -= 1

            ee = ss + len(selected_text)

            # 文頭に空白が一つだけある場合は ee -= 1
            # re.match は文頭から前提っぽい...？
            if re.match(r' [^ ]', row['text']) is not None:
                ee -= 1
            ss = max(0, ss)
            # selected text 以前に '  ' がある場合は
            # selected text が 1 文字以上あり、
            # 後ろから二番目が space である場合は sel = sel[:-2]
            if '  ' in row['text'][:ss] and row['sentiment'] != 'neutral':
                text1 = " ".join(row['text'].split())
                sel = text1[ss:ee].strip()
                if len(sel) > 1 and sel[-2] == ' ':
                    sel = sel[:-2]
                selected_text = sel
            row['fixed_selected_text'] = selected_text
            if row['textID'] in self.map_tkm_st:
                selected_text = self.map_tkm_st[row['textID']]
        else:
            selected_text = row['selected_text']

        # tweet = " " + " ".join(row['text'].split()).lower()
        # selected_text = " " + " ".join(row['selected_text'].split()).lower()
        if self.tokenize_period:
            # tweet_base = re.sub(r'\.', ' %%', row['text'])
            # tweet_base = re.sub('!', ' ##', tweet_base)
            tweet_base = re.sub(r' \.', '[S][PERIOD]', row['text'])
            tweet_base = re.sub(r'\.', '[PERIOD]', tweet_base)
            tweet_base = re.sub(' !', '[S][EXCL]', tweet_base)
            tweet_base = re.sub('!', '[EXCL]', tweet_base)
            # tweet_base = re.sub(' \?', '[S][QUES]', tweet_base)
            # tweet_base = re.sub('\?', '[QUES]', tweet_base)
            tweet = " " + " ".join(tweet_base.split())
            # selected_text_base = re.sub(r'\.', ' %%', row['selected_text'])
            # selected_text_base = re.sub('!', ' ##', selected_text_base)
            selected_text_base = re.sub(r' \.', '[S][PERIOD]', selected_text)
            selected_text_base = re.sub(r'\.', '[PERIOD]', selected_text_base)
            selected_text_base = re.sub(' !', '[S][EXCL]', selected_text_base)
            selected_text_base = re.sub('!', '[EXCL]', selected_text_base)
            # selected_text_base = re.sub(' \?', '[S][QUES]', selected_text_base)
            # selected_text_base = re.sub('\?', '[QUES]', selected_text_base)
            selected_text = " " + " ".join(selected_text_base.split())
        else:
            tweet = " " + " ".join(row['text'].split())
            if self.use_magic:
                selected_text = " ".join(selected_text.split())
            else:
                selected_text = " " + " ".join(selected_text.split())
            # selected_text = " " + " ".join(selected_text.split())

        if self.use_magic:
            idx = tweet.find(selected_text)
            char_targets = np.zeros((len(tweet)))
            char_targets[idx:idx+len(selected_text)] = 1
            if tweet[idx - 1] == ' ':
                char_targets[idx - 1] = 1
        else:
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
                if self.use_magic and tweet[idx0 - 1] == ' ':
                    char_targets[idx0 - 1] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets

        target_idx = []
        best_idx, best_sm = None, -1  # NOTE: this should be 0
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if self.use_magic:
                sm = np.mean(char_targets[offset1: offset2])
                if sm > 0.5 and char_targets[offset1] != 0:
                    target_idx.append(j)
                if sm > best_sm:
                    best_sm = sm
                    best_idx = j
            else:
                if sum(char_targets[offset1: offset2]) > 0:
                    target_idx.append(j)
        # raise best_sm > 0

        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except Exception as e:
            print('=========================')
            print(f'char_targets: {char_targets}')
            # for j, (offset1, offset2) in enumerate(tweet_offsets):
            #     if self.use_magic:
            #         sm = np.mean(char_targets[offset1: offset2])
            #         print(f'sm: {sm}')
            #         print(f'word: {tweet[offset1: offset2]}')
            #         if sm > 0.5 and char_targets[offset1] != 0:
            #             target_idx.append(j)
            # print('-------------------------')
            print(f'row: {row}')
            print(f'tweet: :{tweet}:')
            print(f'selected_text: :{selected_text}:')
            print(f'target_idx: {target_idx}')
            targets_start = best_idx
            targets_end = best_idx

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

        # for MLM
        mlm_input_ids = np.asarray(input_ids)
        mlm_labels = (np.ones(len(input_ids)) * -100).astype(int)
        mask_indices = np.random.choice(np.arange(4, 4+len(input_ids_orig)), int(len(input_ids_orig) * 0.15) + 1)
        for mask_index in mask_indices:
            mlm_labels[mask_index] = mlm_input_ids[mask_index]
            mlm_input_ids[mask_index] = 50264
        row['mlm_input_ids'] = mlm_input_ids
        row['mlm_labels'] = mlm_labels

        # for i in range(len(input_ids)):
        #     if input_ids[i] == 50266:
        #         input_ids[i] = 479
        #     if input_ids[i] == 50267:
        #         input_ids[i] = 27785
        #     if input_ids[i] == 50268:
        #         input_ids[i] = 17487
        row['input_ids'] = input_ids
        row['attention_mask'] = mask
        row['token_type_ids'] = token_type_ids
        row['labels_head'] = targets_start
        if self.tail_index == 'natural':
            row['labels_tail'] = targets_end + 1
        elif self.tail_index == 'kernel':
            row['labels_tail'] = targets_end
        else:
            raise NotImplementedError()
        row['offsets'] = tweet_offsets

        return row


class TSEHeadTailDatasetV4(TSEDataset):
    '''
    use kernal text preprocess logic
    https://www.kaggle.com/abhishek/roberta-inference-5-folds

    label is not int, but segmentation style

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
        labels_head = np.zeros(len(input_ids))
        labels_head[targets_start:] = 1
        row['labels_head'] = labels_head
        labels_tail = np.zeros(len(input_ids))
        labels_tail[targets_end + 1:] = 1
        row['labels_tail'] = labels_tail
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


class TSEHeadTailSegmentationDatasetV3(TSEHeadTailDatasetV3):

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        row = self._prep_text(row)

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']),
            'offsets': torch.tensor(row['offsets']).long(),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']),
            # 'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
            'selected_text': row['selected_text'],
            'labels_head': torch.tensor(row['labels_head']),
            'labels_tail': torch.tensor(row['labels_tail']),
            'labels_segmentation': torch.tensor(row['labels_segmentation']),
            'labels_single_word': torch.tensor(row['labels_single_word']),
            'mlm_input_ids': torch.tensor(row['mlm_input_ids']),
            'mlm_labels': torch.tensor(row['mlm_labels']),
        }

    def _prep_text(self, row):
        row = super()._prep_text(row)
        labels_segmentation = np.zeros(self.max_length)
        if row['labels_head'] >= 0 and row['labels_tail'] >= 0:
            if self.tail_index == 'natural':
                labels_segmentation[row['labels_head']:row['labels_tail']] = 1
            elif self.tail_index == 'kernel':
                labels_segmentation[row['labels_head']:row['labels_tail'] + 1] = 1
        row['labels_segmentation'] = labels_segmentation
        if self.tail_index == 'kernel':
            if row['labels_head'] == row['labels_tail']:
                labels_single_word = row['labels_head']
            else:
                labels_single_word = 0
        else:
            if row['labels_head'] + 1 == row['labels_tail']:
                labels_single_word = row['labels_head']
            else:
                labels_single_word = 0
        row['labels_single_word'] = labels_single_word
        # pad_token = 1
        # pad_token = self.tokenizer.encode_plus(
        #     ' ' + self.tokenizer.special_tokens_map['pad_token'],
        #     text_pair=None,
        #     add_special_tokens=False,
        #     max_length=1)['input_ids'][0]
        # row['labels_segmentation'][np.asarray(
        #     row['input_ids']) == pad_token] = -1
        return row


class TSEHeadTailSegmentationDatasetV4(TSEHeadTailDatasetV3):
    '''
    w/o ignore -1

    '''

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        row = self._prep_text(row)

        return {
            'textID': row['textID'],
            'text': row['text'],
            'input_ids': torch.tensor(row['input_ids']),
            'offsets': torch.tensor(row['offsets']).long(),
            'sentiment': row['sentiment'],
            'attention_mask': torch.tensor(row['attention_mask']),
            # 'special_tokens_mask': torch.tensor(row['special_tokens_mask']).long(),
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
        # pad_token = 1
        # row['labels_segmentation'][np.asarray(
        #     row['input_ids']) == pad_token] = -1
        # pad_token = self.tokenizer.encode_plus(
        #     self.tokenizer.special_tokens_map['pad_token'],
        #     text_pair=None,
        #     add_special_tokens=False,
        #     max_length=1)['input_ids'][0]
        # row['labels_segmentation'][np.asarray(
        #     row['input_ids']) != pad_token] = -1
        return row
