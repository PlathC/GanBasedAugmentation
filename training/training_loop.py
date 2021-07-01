import copy
import os
import time
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict


def training_loop(model: torch.nn.Module,
                  dataloaders: Dict[str, torch.utils.data.DataLoader],
                  criterion: Callable,
                  optimizer: torch.optim.Optimizer,
                  output_dir: str,
                  scheduler: torch.optim.lr_scheduler = None,
                  starting_epoch: int = 0,
                  num_epochs: int = 25,
                  save_best: bool = False,
                  device: torch.device = torch.device('cuda:0')):
    """
    Args:
        model: The model to train
        dataloaders: Dictionary of dataloader which needs to contain 'train' and 'val' loaders
        criterion: Model's loss
        optimizer: Model's optimizer
        output_dir: Log output dir
        scheduler: Training scheduler see https://pytorch.org/docs/stable/optim.html
        starting_epoch: Starting id of the epoch. Useful when starting from a checkpoint
        num_epochs: Number of epoch to do (The real number of epoch done will be num_epochs - starting_epoch)
        save_best: Save checkpoints every new progress
        device: CPU/GPU

    Returns:
        Model with best weights, and the validation accuracy history
    """
    since = time.time()

    val_acc_history = []
    writer = SummaryWriter(os.path.join(output_dir, 'runs'))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(starting_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            saving = True
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print(f'Val accuracy upgrade from {best_acc} to {epoch_acc}, saving...')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                if save_best:
                    saving = False
                    saver = {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'opt': optimizer
                    }

                    if scheduler:
                        saver['scheduler'] = scheduler.state_dict()
                    torch.save(saver, os.path.join(output_dir, f'model_{epoch}_{best_acc}.pt'))

            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if saving:
                saver = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'opt': optimizer
                }

                if scheduler:
                    saver['scheduler'] = scheduler.state_dict()
                torch.save(saver, os.path.join(output_dir, f'model_{epoch}.pt'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
