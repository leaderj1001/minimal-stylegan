import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils

from config import load_args
from preprocess import load_data
from model import StyleGAN

import math
import os
import cv2


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images


def main(args):
    path = 'checkpoint/train_step-6.model'
    generator = StyleGAN()
    if args.cuda:
        generator = generator.cuda()
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['g_running'])
    generator.eval()

    args.size = 128
    step = int(math.log2(args.size)) - 2

    mean_style = get_mean_style(generator, 'cuda')

    with torch.no_grad():
        latent_z = torch.randn(36, 512)
        if args.cuda:
            latent_z = latent_z.cuda()
        img = generator(latent_z, step=step, alpha=1, mean_style=mean_style, style_weight=0.7)
        utils.save_image(img, 'eval.png', nrow=6, normalize=True, range=(-1, 1))

    for j in range(20):
        img = style_mixing(generator, step, mean_style, 5, 3, 'cuda')
        utils.save_image(
            img, f'sample_mixing_{j}.png', nrow=5 + 1, normalize=True, range=(-1, 1)
        )


if __name__ == '__main__':
    args = load_args()
    main(args)