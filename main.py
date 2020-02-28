from fashion import Fashion
import torch
from torch import nn
from torch.autograd import Variable
import utils
from models import simpleCNN, shufflenet, mixnet
import os
import csv
from focalloss import FocalLoss


args = utils.parse_command()
print(args)

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if isinstance(optimizer, torch.optim.SGD):
        lr = lr_init * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print ("lr set to ", lr)

def main():
    train_dataset = Fashion(root="./FashionMNIST", train=True, transform=True, download=True, transform_intensity=args.aug_int)
    test_dataset = Fashion(root="./FashionMNIST", train=False, transform=True, download=True, transform_intensity=args.aug_int)

    out_dir = utils.get_output_directory(args)
    utils.make_dir(out_dir)
    test_csv = os.path.join(out_dir, 'test.csv')
    with open(test_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch','training_loss', 'test_accuracy'])
        writer.writeheader()

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print("=> creating Model ({}-{}-{}) ...".format(args.arch, args.optimizer, args.criterion))
    if args.arch == 'conv':
        model = simpleCNN()
    elif args.arch == 'shufflenet':
        model = shufflenet()
    elif args.arch == 'mixnet':
        model = mixnet()

    if torch.cuda.is_available():
        model.cuda()
    print("=> model created.")

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'fl':
        criterion=FocalLoss(gamma=2)
    #total_iters = args.epochs * (len(train_dataset) // args.batch_size)

    for epoch in range(args.epochs):

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        total_len = len(train_loader)
        model.train()
        if epoch%10==0:
            adjust_learning_rate(optimizer, epoch, args.learning_rate)
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            if args.normalise:
                images=images/255.0
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i%100==0:
                print('TRAIN - Epoch: {} Iter: {}/{} Loss: {} '.format(epoch, i, total_len, loss.data))

        # store latest model
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'optimizer' : optimizer,
        }, epoch, out_dir)

        # test accuracy after each epoch
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for image, label in test_loader:
                if torch.cuda.is_available():
                    image = Variable(image.cuda())
                else:
                    image = Variable(image)
                if args.normalise:
                    image = image / 255.0
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == label.cpu()).sum()
                else:
                    correct += (predicted == label).sum()
        accuracy = 100 * (correct.item() / total)
        print('TEST - Epoch: {} Loss: {} Accuracy: {}'.format(epoch, loss.data, accuracy))

        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch','training_loss', 'test_accuracy'])
            writer.writerow({'epoch': epoch, 'training_loss':loss.data.item(), 'test_accuracy': accuracy})

if __name__ == '__main__':
    main()