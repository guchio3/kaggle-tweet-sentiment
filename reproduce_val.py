import os
import random
from glob import glob

import numpy as np
import pandas as pd
import yaml

import torch
from tools.args import parse_args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(71)


CONFIG_DIR = './configs/exp_configs/'


def reproduce_selected_text(row):
    text = ' '.join(row['text'].split())
    selected_text = ' '.join(row['selected_text'].split())
    offset = text.lower().find(selected_text.lower())
    reproduced_selected_text = text[offset:offset+len(selected_text)]
    if len(reproduced_selected_text) == 0:
        print('-------------------------------------')
        print(f'offset: {offset}')
        print(f'text:{text.lower()}')
        print(f'selected_text:{selected_text.lower()}')
        print(row)
        print('-------------------------------------')
        raise Exception()
    return reproduced_selected_text


if __name__ == '__main__':
    args = parse_args(None)
    exp_id = args.exp_id
    checkpoint = args.checkpoint
    device = args.device
    debug = args.debug
    with open(f'{CONFIG_DIR}/{exp_id}.yml', 'r') as fin:
        config = yaml.load(fin, Loader=yaml.SafeLoader)
    with open(f'{CONFIG_DIR}/e000.yml', 'r') as fin:
        default_config = yaml.load(fin, Loader=yaml.SafeLoader)

    if config['runner'] == 'r001':
        from tools.runners import r001SegmentationRunner as Runner
    elif config['runner'] == 'r002':
        from tools.runners import r002HeadTailRunner as Runner
    elif config['runner'] == 'r003':
        from tools.runners import r003HeadTailSegmentRunner as Runner
    elif config['runner'] == 'r004':
        from tools.runners import r004HeadAnchorRunner as Runner
    else:
        raise NotImplementedError(f'{config["runner"]} is not implemented.')

    trn_df = pd.read_csv('./inputs/datasets/abhishek_folds/train_folds.csv')
    runner = Runner(exp_id, checkpoint, device, debug, config, default_config)
    ckpts = glob(f'./checkpoints/{exp_id}/best/*.pth')

    for ckpt in ckpts:
        fold = int(ckpt.split('_')[1])
        fold_val_df = trn_df.query(f'kfold == {fold}').dropna()
        fold_val_df.to_csv('./temp.csv', index=False)
        textIDs, predicted_texts = runner.predict('./temp.csv', [ckpt])
        trn_df = trn_df.set_index('textID')
        trn_df.loc[textIDs, 'pred_selected_text'] = predicted_texts
        trn_df = trn_df.reset_index()
    if not os.path.exists(f'./inputs/nes_info/pseudo/{exp_id}'):
        os.mkdir(f'./inputs/nes_info/reproduce_vals/{exp_id}')
    trn_df.to_csv(
        f'./inputs/nes_info/reproduce_vals/{exp_id}/reproduced_val.csv',
        index=False)
