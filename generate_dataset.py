import argparse
import csv
import json
import os
import pathlib
from PIL import Image
import shutil
import sys
import torch
from typing import Dict

from network import StyleGenerator, PairsManagerGenerator

ACCEPTED_EXTENSION = [
    '.jpg', '.jpeg', '.png'
]


def get_repartition(split_file) -> Dict:
    """
    Parse a repartition file with a format similar as splitting file in Hyper-Kvasir
    Args:
        split_file: repartition file with a format similar as splitting file in Hyper-Kvasir

    Returns:
        A dictionary associating file paths and 'train' or 'val' values.
    """
    file_repartition = {}
    with open(split_file, 'r') as stream:
        reader = csv.reader(stream, delimiter=';')
        next(reader, None)  # skip the headers
        for row in reader:
            file_name, _, split_index = row
            file_name = file_name.replace('\\', '/')
            file_repartition[os.path.basename(file_name)] = 'train' if int(split_index) == 0 else 'val'
    return file_repartition


if __name__ == '__main__':
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), 'dnnlib'))
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), 'torch_utils'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, nargs='+',
                        default=['./checkpoints/2.sg2ada_non_pathological.pkl',
                                 './checkpoints/3.sg2ada_pathological.pkl'],
                        help='List of string contains paths to SG2 checkpoints (optional if generate_number = 0  '
                             'or one per class defined in class_names) ')
    parser.add_argument('--class_folders', type=json.loads, default='{}',
                        help='String dictionary based on the following format (Keys must be the same as class_names): '
                             '\'{"class_name": ["folder1", ...]}\'')
    parser.add_argument('--generate_number', type=int,
                        help='With dataset equalisation: #(Real Images) + #(Synthetic image) = generate_number, '
                             'Without: #(Synthetic image) = generate_number')
    parser.add_argument('--image_pairs', type=int, default=0,
                        help='Generate only image pairs, if true, indicate the mixed level')
    parser.add_argument('--dataset_equalization', type=bool, default=False,
                        help='Change generation behaviour by fixing the final per class image number')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--split_file', type=str,
                        help='CSV file describing (file-name;class-name;split-index) training (0) and validation sets '
                             '(!=0).')
    args = parser.parse_args()

    print(f'Reading {args.split_file}...')
    repartition = get_repartition(args.split_file)
    print(f'Repartition content read !')

    assert (len(args.checkpoints) == len(args.class_folders.keys())) or (args.generate_number == 0),\
           'There should be the same number of checkpoints and folders when generate_number > 0.'

    class_names = args.class_folders.keys()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Creating gans...')
    if args.generate_number > 0:
        gans = {}
        if args.image_pairs > 0:
            assert len(args.checkpoints) == len(class_names) == 2, 'For a paired generation, only two class can be ' \
                                                                   'provided '
            paired_generator = PairsManagerGenerator(args.checkpoints[0], args.checkpoints[1], args.generate_number)
            gans[class_names[0]] = lambda index: paired_generator.content_image_by_index(index)
            gans[class_names[1]] = lambda index: paired_generator.mixed_image_by_index(index, args.image_pairs)
        else:
            for path, class_name in zip(args.checkpoints, class_names):
                if not os.path.exists(path):
                    print(f'{path} does not exist. No augmentation for class {class_name}')
                    continue
                gans[class_name] = StyleGenerator(path).to(device)
                gans[class_name].eval()

    print('Gans created !')
    print('Producing dataset.')
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.mkdir(args.output_dir)
    os.mkdir(os.path.join(args.output_dir, 'train'))
    os.mkdir(os.path.join(args.output_dir, 'val'))

    for class_name in class_names:
        print(f'Producing data for class "{class_name}"...')

        output_paths = [os.path.join(args.output_dir, mode, class_name) for mode in ['train', 'val']]
        for path in output_paths:
            if not os.path.exists(path):
                os.mkdir(path)

        count = 0
        for folder in args.class_folders[class_name]:
            for path, dirs, files in os.walk(folder):
                for file in files:
                    extension = os.path.splitext(file)[1]
                    if extension in ACCEPTED_EXTENSION:
                        shutil.copyfile(
                            os.path.join(path, file),
                            os.path.join(args.output_dir, repartition[os.path.basename(file)], class_name,
                                         os.path.basename(file))
                        )
                        count += 1

        if args.dataset_equalization:
            generate_count = args.generate_number - count
        else:
            generate_count = args.generate_number

        print(f'{count} real images of class "{class_name}", done.')
        generate_synthetic_images = False
        if generate_count > 0:
            generate_synthetic_images = class_name in gans

        if not generate_synthetic_images:
            print(f'Skip generation since generate_count = {generate_count} or the SG2 checkpoint for {class_name} '
                  f'has not been provided')
            continue

        print('Generation...')
        for i in range(generate_count):
            if args.image_pairs:
                image = gans[class_name](i)
            else:
                image = gans[class_name](gans[class_name].produce_noise(1, device=device))

            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            Image.fromarray(image[0].cpu().numpy(), 'RGB').save(
                os.path.join(args.output_dir, 'train', class_name, str(count) + '.png'))

            count += 1
        print(f'Generation of {generate_count} synthetic images of class "{class_name}", done.')
        print('=' * 10)
