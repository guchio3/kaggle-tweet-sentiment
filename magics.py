import math
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import tokenizers

DIR = "./inputs/datasets/tkm/"

MAX_LEN = 120
# PATH = '../input/tf-roberta/'
PATH = './inputs/datasets/roberta/tokenizer/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    # vocab_file=PATH + 'vocab-roberta-base.json',
    # merges_file=PATH + 'merges-roberta-base.txt',
    vocab_file=PATH + 'vocab.json',
    merges_file=PATH + 'merges.txt',
    lowercase=True,
    add_prefix_space=True
)
SEED = 88888
np.random.seed(SEED)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}


def proc(train):
    # 前処理
    ct = train.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
    start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

    text = train['text'].values
    selected_text = train['selected_text'].values
    sentiments = train['sentiment'].values
    for k in tqdm(range(train.shape[0])):
        ss = text[k].find(selected_text[k])
        # selected text の前に space が 1 or 2 個あったらそれに合わせる
        if text[k][max(ss - 2, 0):ss] == '  ':
            ss -= 2
        if ss > 0 and text[k][ss - 1] == ' ':
            ss -= 1

        ee = ss + len(selected_text[k])

        # 文頭に空白が一つだけある場合は ee -= 1
        # re.match は文頭から前提っぽい...？
        if re.match(r' [^ ]', text[k]) is not None:
            ee -= 1
        ss = max(0, ss)
        # selected text 以前に '  ' がある場合は
        # selected text が 1 文字以上あり、
        # 後ろから二番目が space である場合は sel = sel[:-2]
        if '  ' in text[k][:ss] and sentiments[k] != 'neutral':
            text1 = " ".join(text[k].split())
            sel = text1[ss:ee].strip()
            if len(sel) > 1 and sel[-2] == ' ':
                sel = sel[:-2]

            selected_text[k] = sel
        # selected_text[k] = re.sub('[^AaIiUu] ', '', selected_text[k])
        # FIND OVERLAP
        text1 = " " + " ".join(text[k].split())
        text2 = " ".join(selected_text[k].split())
        idx = text1.find(text2)

        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ':
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)

        # ID_OFFSETS
        offsets = enc.offsets

        # START END TOKENS
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.mean(chars[a:b])
            if sm > 0.5 and chars[a] != 0:
                toks.append(i)

        s_tok = sentiment_id[train.loc[k, 'sentiment']]
        input_ids[k, :len(enc.ids) + 3] = [0, s_tok] + enc.ids + [2]
        attention_mask[k, :len(enc.ids) + 3] = 1
        if len(toks) > 0:
            start_tokens[k, toks[0] + 2] = 1
            end_tokens[k, toks[-1] + 2] = 1
    train.to_csv('train_new.csv', index=False)
    return (input_ids,
            attention_mask,
            token_type_ids,
            start_tokens,
            end_tokens)


# @jit
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def load_data():
    train = pd.read_csv(
        './inputs/origin/train.csv').fillna('')
        # './input/tweet-sentiment-extraction/train.csv').fillna('')
    print(train.head())

    data = proc(train)
    with open('data2.pkl', 'wb') as f:
        pickle.dump(data, f, -1)


def main():
    train = pd.read_csv(
        './inputs/origin/train.csv').fillna('')
        # '../input/tweet-sentiment-extraction/train.csv').fillna('')
    text = train['text'].values
    selected_text = train['selected_text'].values

    # proc() を参照
    # with open('data2.pkl', 'rb') as f:
    #    data = pickle.load(f)
    # (input_ids,
    # attention_mask,
    # token_type_ids,
    # start_tokens,
    # end_tokens) = data

    # CVの予測結果 n samples x MAX_LEN
    oof_start = np.load(DIR + 'oof_start.npy')
    oof_end = np.load(DIR + 'oof_end.npy')
    oof_all = np.load(DIR + 'oof_all.npy')  # 私のモデルでは一応セグメンテーションも予測している

    i = 0
    list_st = []

    all = []
    for k in range(oof_start.shape[0]):
        if 'neutral' == train.loc[k, 'sentiment']:
            st = text[k].strip().lower()
        else:
            text1 = " " + " ".join(text[k].split())

            enc = tokenizer.encode(text1)

            aa = np.argmax(oof_start[k])
            bb = np.argmax(oof_end[k])
            # head tail 反転に segmentation を利用
            if aa > bb:
                idx = oof_all[k] >= 0.5
                if idx.sum() > 0:
                    idx = np.arange(oof_all.shape[1])[idx]
                    aa = idx[0]
                    bb = idx[-1]
                else:
                    aa = bb = oof_all[k].argmax()

            text0 = text[k]
            ss = 0 if aa - 2 == 0 else enc.offsets[aa - 2][0]

            # NOTE: なんで +1 ?
            if bb - 2 >= len(enc.offsets) - 1:
                ee = enc.offsets[-1][1] + 1
            else:
                ee = enc.offsets[bb - 2][1] + 1

            st = text1[ss:ee].strip()

            ee -= text0[ss:ee].strip().count('   ')
            ee += text0[ss:ee].strip().count('  ')

            if '  ' in text0[:(ss + ee) // 2]:
                st = text0[ss:ee].strip()

        list_st.append(st)
        sc = jaccard(st, selected_text[k])

        all.append(sc)
    print(i, '>>>> FOLD Jaccard =', np.mean(all))


if __name__ == '__main__':
    load_data()
