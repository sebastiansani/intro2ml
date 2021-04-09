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


chkpt_dir = 'chkpts'
dataset_root = 'dataset'
n_epoch = 50
best_eval_loss = 9999
batch_size = 32

# get network class names
net_class_names = get_network_class_names()

# parse arguments
parser = argparse.ArgumentParser(description='train model')
parser.add_argument('-m', '--model', default='vgg_pretrained',
                    help='model to train', type=str, choices=net_class_names)
parser.add_argument(
    '-s', '--save', help='create chkpts and log files', action='store_true')
args = parser.parse_args()

log_progress = args.save  # create chkpts and log files

print('-'*30)
print('training {}'.format(args.model))
if log_progress:
    print('model is being saved')
else:
    print('model is NOT being saved, run --save to generate chkpts and log files')

dataset = Dataset(dataset_root, set_seed=True)
dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)

# get model
class_ = getattr(network, args.model)
net = class_()

net.cuda()
net.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12)

if not os.path.isdir(chkpt_dir):
    os.mkdir(chkpt_dir)

eval_loss = np.empty((dataset.get_eval_length(),))

start_time = time.time()
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

            prediction = torch.argmax(output)
            guess_count[prediction] += 1
            if prediction == label:
                accuracy += 1
                correct_guesses[prediction] += 1

            eval_loss[j] = criterion(output, label)

        class_wise_acc = sum([correct_guesses[i]/guess_count[i]
                              if guess_count[i] != 0 else 0 for i in range(10)])/10

        accuracy = accuracy/dataset.get_eval_length()

        print('e{:02d} - train_loss {:.4f} | valid_loss {:.4f} | accuracy {:.4f} | class_wise_acc {:.4f}'.format(
            epoch, loss.item(), np.mean(eval_loss), accuracy, class_wise_acc))

        # save best chkpt
        if log_progress and np.mean(eval_loss) < best_eval_loss:
            fckpt_name = '{}/MODEL1_BEST.pth'.format(
                chkpt_dir)
            torch.save(net.state_dict(), fckpt_name)
            best_eval_loss = np.mean(eval_loss)

    net.train()

    scheduler.step()

end_time = time.time()
print('-'*30)
print('end', time.ctime())
print('total', timedelta(seconds=end_time-start_time))