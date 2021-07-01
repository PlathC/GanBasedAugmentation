import argparse
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import tqdm

import utils
from data import CustomDataset, SpecififRotateTransform
from network import create_classifier
from training import training_loop


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Training dataset folder path')
    parser.add_argument('--input_size', type=int, default=256, help='Default image size')
    parser.add_argument('--da', type=bool, default=False, help='Enable data augmentation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate value')
    parser.add_argument('--clr', type=bool, default=True, help='Enable cyclic learning rate')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epoch_number', type=int, default=30, help='Epoch nb')
    parser.add_argument('--pretrained', type=str, default='', help='Load pretrained densenet (ImageNet or path to the '
                                                                   'checkpoint file)')
    parser.add_argument('--save_best', type=bool, default=False, help='Save each model epoch that outperforms the '
                                                                      'previous ones')
    parser.add_argument('--continue_train', type=str, default='', help='Continue training from checkpoint')
    parser.add_argument('--architecture', type=str, default='densenet161', help='Classifier architecture')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets = {}
    dataloaders = {}
    for mode in ['train', 'val']:
        data = {}
        for folder in os.listdir(os.path.join(args.dataset, mode)):
            if folder not in data:
                data[folder] = []
            data[folder] += [os.path.join(args.dataset, mode, folder)]

        if mode == 'train' and args.da:
            datasets[mode] = CustomDataset(data, transform=transforms.Compose([
                transforms.Resize(args.input_size),
                transforms.ColorJitter(hue=.05, saturation=.05),
                SpecififRotateTransform((0, 90, 180, 270)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0.1, 0.), shear=8),
                transforms.RandomCrop(args.input_size),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
        else:
            datasets[mode] = CustomDataset(data, transform=transforms.Compose([
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        dataloaders[mode] = DataLoader(datasets[mode], args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    class_nb = len(datasets['train'].classes)
    network = create_classifier(args.architecture, class_nb, pretrained=args.pretrained == 'ImageNet').to(device)
    if args.pretrained != '' and args.pretrained != 'ImageNet':
        print("Using pre-trained model '{}'".format(args.pretrained))

        saver = torch.load(args.pretrained)
        pretrained_state = saver['model']
        model_state = network.state_dict()

        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        network.load_state_dict(model_state)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=0.9)

    scheduler = None
    if args.clr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr,
            max_lr=0.01,
        )

    starting_epoch = 0
    if args.continue_train:
        saver = torch.load(args.continue_train)
        network.load_state_dict(saver['model'])
        optimizer = saver['opt']
        starting_epoch = saver['epoch']

        if 'scheduler' in saver and args.clr:
            scheduler.load_state_dict(saver['scheduler'])

    model, _ = training_loop(
        network,
        dataloaders,
        criterion,
        optimizer,
        args.output_dir,
        scheduler=scheduler,
        starting_epoch=starting_epoch,
        num_epochs=args.epoch_number,
        save_best=args.save_best,
        device=device
    )

    torch.save({
        'model': model
    }, os.path.join(args.output_dir, 'model_full.pt'))

    running_corrects = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
    with torch.no_grad():
        for inputs, classes in tqdm(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            all_labels_d = torch.cat((all_labels_d, classes), 0)
            all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
            all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)

            running_corrects += torch.sum(predicted == classes.data)
    epoch_acc = running_corrects.double() / len(datasets['val'])
    print(f'Accuracy : {epoch_acc}')

    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    testset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')

    print(datasets['val'].classes)
    results = utils.metrics(y_true, y_predicted, datasets['val'].classes)
    print(results)

    out_file = os.path.join(args.output_dir, 'results.json')
    with open(out_file, 'w') as f:
        f.write(json.dumps(results, cls=NumpyEncoder))
    print(f'Metrics written in {out_file}')
