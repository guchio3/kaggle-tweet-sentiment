import argparse


def parse_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python -e e001 -d -m "e001, basic experiment"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-e', '--exp_id',
                        help='experiment setting',
                        type=str,
                        required=True)
    parser.add_argument('-c', '--checkpoint',
                        help='the checkpoint u use',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('-d', '--device',
                        help='cpu or cuda, the device for running the model',
                        type=str,
                        required=False,
                        default='cuda')
    parser.add_argument('--debug',
                        help='whether or not to use debug mode',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    return args
