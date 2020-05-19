import os
import yaml
import random
import numpy as np
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
    else:
        raise NotImplementedError(f'{config["runner"]} is not implemented.')
    runner = Runner(exp_id, checkpoint, device, debug, config, default_config)
    runner.train()
