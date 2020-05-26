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
    text = row['text']
    selected_text = row['selected_text'][1:]  # remove head space
    offset = text.lower().find(selected_text)
    reproduced_selected_text = text[offset:offset+len(selected_text)]
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

    tst_df = pd.read_csv('./inputs/origin/test.csv')
    runner = Runner(exp_id, checkpoint, device, debug, config, default_config)
    ckpts = glob(f'./checkpoints/{exp_id}/best/*.pth')
    for ckpt in ckpts:
        textIDs, predicted_texts = runner.predict(
            './inputs/origin/test.csv', [ckpt])
        tst_df = tst_df.set_index('textID')
        tst_df.loc[textIDs, 'selected_text'] = predicted_texts
        tst_df = tst_df.reset_index()
        tst_df['selected_text'] = tst_df.apply(reproduce_selected_text, axis=1)
        if not os.path.exists(f'./inputs/nes_info/pseudo/{exp_id}'):
            os.mkdir(f'./inputs/nes_info/pseudo/{exp_id}')
        tst_df.to_csv(
            f'./inputs/nes_info/pseudo/{exp_id}/{ckpt.split("/")[-1][:-4]}_pseudo.csv',
            index=False)

    # predict using all
    textIDs, predicted_texts = runner.predict(
        './inputs/origin/test.csv', ckpts)
    tst_df = tst_df.set_index('textID')
    tst_df.loc[textIDs, 'selected_text'] = predicted_texts
    tst_df = tst_df.reset_index()
    tst_df['selected_text'] = tst_df.apply(reproduce_selected_text, axis=1)
    tst_df.to_csv(
        f'./inputs/nes_info/pseudo/{exp_id}/ensembled_pseudo.csv',
        index=False)
