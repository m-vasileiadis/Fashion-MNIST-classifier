import torch
import os
import errno

def parse_command():
    model_names = ['conv', 'shufflenet', 'mixnet']
    loss_names = ['ce', 'fl']
    optimizer_names = ['sgd', 'adam']

    import argparse
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='conv', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: conv)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='ce', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: ce)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('-op', '--optimizer', default='sgd', choices=optimizer_names,
                        help='optmizer: ' + ' | '.join(optimizer_names) + ' (default: sgd)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--normalise', action='store_true', help='normalise to [0-1]')
    parser.add_argument('--aug_int', action='store_true', help='image intensity augmentation')
    args = parser.parse_args()
    return args

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def get_output_directory(args):
    output_directory = os.path.join('results',
                                    'arch={}.criterion={}.lr={}.bs={}.optimizer={}'.
                                    format(args.arch, args.criterion, args.learning_rate, args.batch_size, args.optimizer))
    return output_directory


