import argparse
import cv2
import os


def prepare(args, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), mode='train'):
    img_path = os.path.join(args.base_dir, mode)
    img_list = os.listdir(img_path)

    out_mode = os.path.join(args.out, mode)
    if not os.path.isdir(out_mode):
        os.mkdir(out_mode)

    for size in sizes:
        size_dir = os.path.join(out_mode, '{}_{}'.format(mode, size))
        if not os.path.isdir(size_dir):
            os.mkdir(size_dir)

        for _ in img_list:
            filename = os.path.join(img_path, _)
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize=(size, size))

            out_filename = os.path.join(size_dir, _)
            cv2.imwrite(out_filename, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='E:/FFHG/prepared_dataset')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--base_dir', type=str, default='E:/FFHG/dataset')

    args = parser.parse_args()

    prepare(args, mode='train')
