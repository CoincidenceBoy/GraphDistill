import argparse


def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--epoch', default=0, type=int, metavar='N', help='num of epoch')

    args = parser.parse_args()
    main(args)
