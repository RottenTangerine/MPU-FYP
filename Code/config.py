import argparse

def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',
                        type=str,
                        default='Dataset/ICPR2018',
                        help='dataset path')

    # parameters
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=20,
                        help='default training epochs')
    parser.add_argument('-l',
                        '--lr',
                        type=float,
                        default=1e-3,
                        help='initial learning rate')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=32,
                        help='batch size')

    # model
    parser.add_argument('--cuda',
                        type=bool,
                        default=False,
                        help='Use GPU training')

    return parser.parse_args()
