# adapted from https://github.com/ashmeet13/FashionMNIST-CNN/blob/master/Fashion.py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import urllib.request
import numpy as np
import random
import struct
import torch
import errno
import gzip
import os


class Fashion(Dataset):
    """Dataset: https://github.com/zalandoresearch/fashion-mnist
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in a numpy image
            and may return a horizontally flipped image."""

    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    ]

    file_name = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]

    raw = "raw"
    processsed = "processsed"

    def __init__(self, root="data", train=True, transform=True, download=False, transform_intensity=False):
        super(Fashion, self).__init__()
        self.root = root
        self.transform = transform
        self.transform_intensity = transform_intensity
        self.train = train
        self.tensor_transform = transforms.ToTensor()

        raw_path = os.path.join(self.root, self.raw)
        if download and (os.path.exists(raw_path) == False):
            self.download(self.root)

        if self.train:
            train_path = os.path.join(self.root, self.processsed, "training_set.pt")
            self.train_images, self.train_labels = torch.load(train_path)
        else:
            test_path = os.path.join(self.root, self.processsed, "testing_set.pt")
            self.test_images, self.test_labels = torch.load(test_path)

    '''
    __getitem__(index) -> Will return the image and label at the specified index

    If transform parametr of class is set as True the function would or would not
    perform a random horizontal flip of the image.
    '''

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_images[index], self.train_labels[index]
        else:
            image, label = self.test_images[index], self.test_labels[index]

        image = image.numpy()
        image = np.rot90(image, axes=(1, 2)).copy()

        if self.transform and self.train:
            image = self.transform_process(image, self.transform_intensity)

        image = self.tensor_transform(image/255.0)
        image = image.contiguous()
        image = image.view(1, 28, 28)


        return image, label

    def __len__(self):
        if self.train:
            return (len(self.train_images))
        else:
            return (len(self.test_images))

    def transform_process(self, image, intensity=False):  # Would or would not return a flipped image
        self.rotate = random.getrandbits(1)
        image = np.flip(image, self.rotate).copy()
        if intensity:
            image=image*random.uniform(0.85, 1.15)
            image=np.clip(image,0,255)
        return image

    '''
    download(root) -> The function will download and save the MNIST images in raw
    format under the 'raw' folder under the user specified root directory
    '''

    def download(self, root):
        raw_path = os.path.join(self.root, self.raw)
        processsed_path = os.path.join(self.root, self.processsed)

        try:
            os.makedirs(raw_path)
            os.makedirs(processsed_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        for file_index in range(len(self.file_name)):
            print("Downloading:", self.urls[file_index])
            urllib.request.urlretrieve(self.urls[file_index], (self.file_name[file_index] + '.gz'))
            print("Extracting:", self.file_name[file_index] + ".gz")
            f = gzip.open(self.file_name[file_index] + '.gz', 'rb')
            with open(raw_path + "/" + self.file_name[file_index], 'wb') as w:
                for line in f.readlines():
                    w.write(line)
            f.close()
            os.remove(self.file_name[file_index] + ".gz")

        print()
        print("Raw data downloaded and extracted in your specified root directory under /raw")
        print()
        self.process(self.root)

    '''
    process(root) -> Will process the raw downloaded files into a usable format
    and store them into the a 'processed' folder under user specified root
    directory.
    '''

    def process(self, root):
        raw_path = os.path.join(self.root, self.raw)
        processsed_path = os.path.join(self.root, self.processsed)

        print("Processing training data")
        train_image = self.readimg(self.root, self.file_name[0], 2051)
        train_label = self.readlab(self.root, self.file_name[1], 2049)
        train_data = (train_image, train_label)

        print("Processing testing data")
        test_image = self.readimg(self.root, self.file_name[2], 2051)
        test_label = self.readlab(self.root, self.file_name[3], 2049)
        test_data = (test_image, test_label)

        train_path = os.path.join(self.root, self.processsed, "training_set.pt")
        with open(train_path, "wb") as f:
            torch.save(train_data, f)

        test_path = os.path.join(self.root, self.processsed, "testing_set.pt")
        with open(test_path, "wb") as f:
            torch.save(test_data, f)
        print()
        print("Processed data has been stored in your specified root directory under /processsed")
        print()

    def readimg(self, root, file, magic):
        image = []
        path = os.path.join(self.root, self.raw, file)
        with open(path, 'rb') as f:
            magic_number, size, row, col = struct.unpack('>IIII', f.read(16))
            assert (magic_number == magic)
            for run in range(size * row * col):
                image.append(list(struct.unpack('B', f.read(1)))[0])
            image = np.asarray(image, dtype=np.float32)
            return (torch.from_numpy(image).view(size, 1, row, col))

    def readlab(self, root, file, magic):
        label = []
        path = os.path.join(self.root, self.raw, file)
        with open(path, 'rb') as f:
            magic_number, size = struct.unpack(">II", f.read(8))
            assert (magic_number == magic)
            for run in range(size):
                label.append(list(struct.unpack('b', f.read(1)))[0])
            label = np.asarray(label)
            return (torch.from_numpy(label))