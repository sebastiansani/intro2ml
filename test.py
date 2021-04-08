import csv
import os
from torchvision import transforms
from PIL import Image
from network import vgg_pretrained
import torch
from classes import classes


def load_image(path):
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


n_files_to_test = 5321
root_dir = 'dataset'
ckpt_name = 'MODEL1_BEST.pth'

test_dir = os.path.join(root_dir, 'test')

net = vgg_pretrained()

checkpoint = '{}/{}'.format('chkpts', ckpt_name)
net.load_state_dict(torch.load(checkpoint))

net.cuda()
net.eval()

csvfile = open(ckpt_name+'_results.csv', 'w', newline='')
spamwriter = csv.writer(csvfile)

for image_idx in range(n_files_to_test):
    image = load_image(os.path.join(test_dir, str(
        image_idx)+'.png')).unsqueeze(0).cuda()
    with torch.no_grad():
        output = net(image).squeeze()
    spamwriter.writerow([image_idx, classes[torch.argmax(output).item()]])

csvfile.close()
