import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--path', type=str, default='../Dataset/ICPR2018', help='dataset path')
    parser.add_argument('--test_split_ratio', type=float, default=0.2,
                        help='test data split ratio from the dataset, 0 < ratio < 1')

    # CTPN
    parser.add_argument('--IOU_NEGATIVE', type=float, default=0.3)
    parser.add_argument('--IOU_POSITIVE', type=float, default=0.7)
    parser.add_argument('--RPN_POSITIVE_NUM', type=int, default=150)
    parser.add_argument('--RPN_TOTAL_NUM', type=int, default=300)
    parser.add_argument('--IMAGE_MEAN', type=list, default=[123.68, 116.779, 103.939])

    # parameters
    parser.add_argument('--epochs', type=int, default=100, help='default training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    # train
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU training')

    return parser.parse_args()
