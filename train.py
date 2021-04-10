import torch
from dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import network
import os
import time
from datetime import timedelta
import argparse
import importlib
from utils import get_network_class_names
from classes import classes


# tweakable parameters
chkpt_dir = 'chkpts'
dataset_root = 'dataset'
n_epoch = 50
batch_size = 32
do_data_augmentation = True
grayscale = False
validation_to_train = 0.05

# get network class names
net_class_names = get_network_class_names()

# parse arguments
parser = argparse.ArgumentParser(description='train model')
parser.add_argument('-m', '--model', default='vgg_pretrained',
                    help='model to train', type=str, choices=net_class_names)
parser.add_argument(
    '-s', '--save', help='create chkpts and log files', action='store_true')
parser.add_argument(
    '-b', '--balance', help='balance dataset classes', action='store_true')
parser.add_argument(
    '-c', '--classacc', help='print class accuracies', action='store_true')
args = parser.parse_args()

log_progress = args.save  # create chkpts and log files
balance_data = args.balance  # balance dataset classes
print_class_acc = args.classacc  # print class accuracies

# create dataset
dataset = Dataset(dataset_root, validation_to_train=validation_to_train,
                  set_seed=True, do_data_augmentation=do_data_augmentation, grayscale=grayscale)

if balance_data:
    # create balanced dataset sampler
    weights = dataset.get_sample_weights()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler)
else:
    # random sampler
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)

# get model
class_ = getattr(network, args.model)
net = class_()

net.cuda()
net.train()

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(net.parameters(), lr=5e-4,
                      weight_decay=1e-5, momentum=0.9)

# scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

# create checkpoint dir
if log_progress and not os.path.isdir(chkpt_dir):
    os.mkdir(chkpt_dir)

best_eval_loss = 9999

start_time = time.time()
print('training', args.model)
print('log progress', log_progress)
print('balance data', balance_data)
print('data aug', do_data_augmentation)
print('grayscale', grayscale)
print('start', time.ctime())
print('-'*30)

total_iterations = 0
for epoch in range(n_epoch):
    '''
    TRAIN
    '''
    for batch, labels in dataloader:
        batch = batch.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        output = net(batch)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        total_iterations += 1

    '''
    VALIDATION
    '''
    net.eval()

    with torch.no_grad():

        eval_loss_total = 0
        accuracy = 0
        guess_count = [0 for i in range(10)]
        correct_guesses = [0 for i in range(10)]

        for j in range(dataset.get_eval_length()):
            image, label = dataset.get_eval_item(j)
            image = image.cuda()
            label = label.cuda()

            image = image.unsqueeze(0)
            label = label.unsqueeze(0)

            output = net(image)

            # count for accuracy + class wise accuracy
            prediction = torch.argmax(output)
            guess_count[prediction] += 1
            if prediction == label:
                accuracy += 1
                correct_guesses[prediction] += 1

            # count eval loss
            eval_loss_total += criterion(output, label).item()

        # calculate eval loss
        eval_loss = eval_loss_total/dataset.get_eval_length()

        # calculate accuracy
        accuracy = accuracy/dataset.get_eval_length()

        # calculate class wise accuracy
        class_wise_acc = sum([correct_guesses[i]/guess_count[i]
                              if guess_count[i] != 0 else 0 for i in range(10)])/10

        # print info
        print('e{:02d} - train_loss {:.4f} | valid_loss {:.4f} | accuracy {:.4f} | class_wise_acc {:.4f}'.format(
            epoch, loss.item(), eval_loss, accuracy, class_wise_acc))

        if print_class_acc:
            for i in range(10):
                print('{:23s} acc: {:.4f}'.format(classes[i], correct_guesses[i]/guess_count[i]
                                                  if guess_count[i] != 0 else 0))

        # save best chkpt
        if log_progress and eval_loss < best_eval_loss:
            ckpt_name = '{}{}_best.pth'.format(
                args.model, '_balanced' if balance_data else '')
            fckpt_path = os.path.join(chkpt_dir, ckpt_name)
            torch.save(net.state_dict(), fckpt_path)
            best_eval_loss = eval_loss

    net.train()

    scheduler.step()

end_time = time.time()
print('-'*30)
print('end', time.ctime())
print('total', timedelta(seconds=end_time-start_time))
