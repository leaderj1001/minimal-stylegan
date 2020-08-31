import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad

from config import load_args
from preprocess import load_data
from model import StyleGAN, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, loader, generator, g_running, discriminator, g_optimizer, d_optimizer):
    import math
    from tqdm import tqdm
    args.max_iteration = 200000
    pbar = tqdm(range(args.max_iteration))
    # train_loader = iter(loader)
    args.phase = 200000 // 5
    step = 5
    resolution = 128

    adjust_lr(g_optimizer, 0.0015)
    adjust_lr(d_optimizer, 0.0015)

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    d_loss_val = 0
    g_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(128)) - 2
    final_progress = False

    global_step = 0
    for i in pbar:
        discriminator.zero_grad()

        if global_step > args.max_iteration:
            break

        alpha = min(1, 1 / args.phase * (used_sample + 1))
        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            if torch.cuda.device_count() > 1:
                torch.save(
                    {
                        'generator': generator.module.state_dict(),
                        'discriminator': discriminator.module.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'g_running': g_running.state_dict(),
                    },
                    f'checkpoint/train_step-{ckpt_step}.model',
                )
            else:
                torch.save(
                    {
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'g_running': g_running.state_dict(),
                    },
                    f'checkpoint/train_step-{ckpt_step}.model',
                )

            adjust_lr(g_optimizer, 0.0015)
            adjust_lr(d_optimizer, 0.0015)
        if i % 99 == 0:
            train_loader = iter(loader)

        try:
            data = next(train_loader)
        except(OSError, StopIteration):
            train_loader = iter(loader)
            data = next(train_loader)
        img = data[str(args.max_size)]
        if args.cuda:
            img = img.cuda()

        used_sample += img.shape[0]

        b_size = img.size(0)

        if args.loss == 'wgan-gp':
            d_real = discriminator(img, step=step, alpha=alpha)
            d_real = d_real.mean() - 0.001 * (d_real ** 2).mean()
            (-d_real).backward()
        elif args.loss == 'r1':
            img.requires_grad = True
            d_scores = discriminator(img, step=step, alpha=alpha)
            d_real = F.softplus(-d_scores).mean()
            d_real.backward(retain_graph=True)

            grad_real = grad(outputs=d_scores, inputs=img, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()

            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()

        gen_in1, gen_in2 = torch.randn(2, b_size, 512, device='cuda').chunk(
            2, 0
        )
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

        gen_img = generator(gen_in1, step=step, alpha=alpha)
        d_gen = discriminator(gen_img, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            d_gen = d_gen.mean()
            d_gen.backward()

            eps = torch.randn(b_size, 1, 1, 1).cuda()
            x_hat = eps * img.data + (1 - eps) * gen_img.data
            x_hat.requires_grad = True
            d_hat = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(d_hat.sum(), x_hat, create_graph=True)[0]
            grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()

            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()
                d_loss_val = (-d_real + d_gen).item()

        else:
            d_gen = F.softplus(d_gen).mean()
            d_gen.backward()

            if i % 10 == 0:
                d_loss_val = (d_real + d_gen).item()

        d_optimizer.step()

        if (i + 1) % args.n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            gen_img = generator(gen_in2, step=step, alpha=alpha)
            pred = discriminator(gen_img, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -pred.mean()
            else:
                loss = F.softplus(-pred).mean()

            if i % 10 == 0:
                g_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Global Step: {global_step}; '
            f'Size: {4 * 2 ** step}; G: {g_loss_val:.3f}; D: {d_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


def main(args):
    train_loader, test_loader = load_data(args)

    generator = StyleGAN()
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    g_running = StyleGAN()
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        g_running = g_running.cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(generator.generator.parameters(), lr=args.lr, betas=(0., 0.99))
    g_optimizer.add_param_group({
        'params': generator.style.parameters(),
        'lr': args.lr * 0.01,
        'mult': 0.01
    })

    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))

    accumulate(g_running, generator, 0)

    train(args, train_loader, generator, g_running, discriminator, g_optimizer, d_optimizer)


if __name__ == '__main__':
    args = load_args()
    main(args)