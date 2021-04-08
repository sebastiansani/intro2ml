import os
import torch
import torch.utils.data as data
import numpy as np
from classes import classes
from torchvision import transforms
from utils import load_image


class Dataset(data.Dataset):
    def __init__(self, root, validation_to_train=0.05, set_seed=False, do_data_augmentation=True):
        if set_seed:
            np.random.seed(12345678)

        self.do_data_augmentation = do_data_augmentation

        self.root = root
        self.train_dir = os.path.join(root, 'train')

        self.train_set = []
        self.valid_set = []

        for train_sub_dir in os.walk(self.train_dir):
            if train_sub_dir[0] != self.train_dir:
                n_files_to_validation = round(
                    validation_to_train*len(train_sub_dir[2]))
                shuffled_files = np.random.permutation(train_sub_dir[2])
                label = classes.index(os.path.basename(
                    os.path.normpath(train_sub_dir[0])))
                files_with_labels = [(file_name, label)
                                     for file_name in shuffled_files]
                self.train_set.extend(
                    files_with_labels[n_files_to_validation:])
                self.valid_set.extend(
                    files_with_labels[:n_files_to_validation])

    def __getitem__(self, index):
        image, label = self.train_set[index]
        image = os.path.join(self.train_dir, classes[label], image)
        image = load_image(image)
        if self.do_data_augmentation:
            data_aug = transforms.Compose(
                [transforms.ColorJitter(),
                 transforms.RandomGrayscale(),
                 transforms.RandomRotation(90),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip()])
            image = data_aug(image)
        return image, label

    def __len__(self):
        return len(self.train_set)

    def get_eval_length(self):
        return len(self.valid_set)

    def get_eval_item(self, index):
        image, label = self.valid_set[index]
        image = os.path.join(self.train_dir, classes[label], image)
        return load_image(image), torch.tensor(label)
