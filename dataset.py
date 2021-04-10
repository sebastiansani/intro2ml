import os
import torch
import torch.utils.data as data
import numpy as np
from classes import classes
from torchvision import transforms
from utils import load_image


class Dataset(data.Dataset):
    def __init__(self, root, validation_to_train=0.05, set_seed=False, do_data_augmentation=True, grayscale=False):
        if set_seed:
            np.random.seed(12345678)

        self.do_data_augmentation = do_data_augmentation
        self.grayscale = grayscale

        self.root = root
        self.train_dir = os.path.join(root, 'train')

        self.train_set = []
        self.valid_set = []

        # walk every subdirectory
        for train_sub_dir in os.walk(self.train_dir):
            if train_sub_dir[0] != self.train_dir:
                # number of files to validate
                n_files_to_validation = round(
                    validation_to_train*len(train_sub_dir[2]))
                # shuffle file order
                shuffled_files = np.random.permutation(train_sub_dir[2])
                # get folder label [0,9]
                label = classes.index(os.path.basename(
                    os.path.normpath(train_sub_dir[0])))
                # pair file name with label
                files_with_labels = [(file_name, label)
                                     for file_name in shuffled_files]
                # add tuples to train set
                self.train_set.extend(
                    files_with_labels[n_files_to_validation:])
                # add tuples to validation set
                self.valid_set.extend(
                    files_with_labels[:n_files_to_validation])

    def get_sample_weights(self):
        '''
        sample-wise weights for uniform class sampling
        '''
        class_counts = [1840, 757, 1311, 1297,
                        996, 1280, 1851, 1419, 1430, 234]
        class_weights = 1. / torch.Tensor(class_counts)
        sample_weights = [class_weights[class_idx]
                          for (_, class_idx) in self.train_set]
        return sample_weights

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        image_name, label = self.train_set[index]
        image_path = os.path.join(self.train_dir, classes[label], image_name)
        image = load_image(image_path, self.grayscale)

        if self.do_data_augmentation:
            data_aug = transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])

            image = data_aug(image)

        return image, label

    def get_eval_length(self):
        return len(self.valid_set)

    def get_eval_item(self, index):
        image, label = self.valid_set[index]
        image = os.path.join(self.train_dir, classes[label], image)
        return load_image(image, self.grayscale), torch.tensor(label)
