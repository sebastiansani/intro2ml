import csv
import os
import torch
from classes import classes
from utils import load_image, get_network_class_names
import network
import argparse
import importlib


n_files_to_test = 5321
results_dir = 'results'
dataset_dir = 'dataset'
checkpoint_dir = 'model'

test_dir = os.path.join(dataset_dir, 'test')

# get network class names
net_class_names=get_network_class_names()

# parse arguments
parser = argparse.ArgumentParser(description='benchmark pretrained model')
parser.add_argument('name', help='checkpoint name')
parser.add_argument('-m', '--model', default='vgg_pretrained', help='model to benchmark', type=str, choices=net_class_names)
args=parser.parse_args()

# get model
class_ = getattr(network, args.model)
net = class_()

ckpt_name = args.name

checkpoint = '{}/{}'.format(checkpoint_dir, ckpt_name)
net.load_state_dict(torch.load(checkpoint))

net.cuda()
net.eval()

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

csv_file_name=ckpt_name.split('.')[0]+'_results.csv'
csv_file_path=os.path.join(results_dir, csv_file_name)

csvfile = open(csv_file_path, 'w', newline='')
spamwriter = csv.writer(csvfile)

for image_idx in range(n_files_to_test):
    image = load_image(os.path.join(test_dir, str(
        image_idx)+'.png')).unsqueeze(0).cuda()
    with torch.no_grad():
        output = net(image).squeeze()
    spamwriter.writerow([image_idx, classes[torch.argmax(output).item()]])

csvfile.close()
