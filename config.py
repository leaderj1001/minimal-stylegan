import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='E:/')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_size', type=list, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument('--max_size', type=int, default=128)
    parser.add_argument('--init_size', type=int, default=8)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='wgan-gp')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )

    args = parser.parse_args()
    return args
