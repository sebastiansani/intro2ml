import os
import torch
import torch.utils.data as data
import numpy as np
from classes import classes
from PIL import Image
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, root, validation_to_train=0.05, set_seed=False):
        if set_seed:
            np.random.seed(12345678)

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

    def load_image(self, path):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        image = Image.open(path)
        image = preprocess(image)
        return image

    def __getitem__(self, index):
        image, label = self.train_set[index]
        image = os.path.join(self.train_dir, classes[label], image)
        return self.load_image(image), label

    def __len__(self):
        return len(self.train_set)

    def get_eval_length(self):
        return len(self.valid_set)

    def get_eval_item(self, index):
        image, label = self.valid_set[index]
        image = os.path.join(self.train_dir, classes[label], image)
        return self.load_image(image), torch.tensor(label)
