import torch
from dataset import Dataset
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np

chkpt_dir='chkpts'
dataset_root = 'dataset'
n_epoch = 100

dataset = Dataset(dataset_root, set_seed=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

net = torchvision.models.vgg19_bn(pretrained=True, progress=False)
net.classifier[6]=nn.Linear(4096,10)

net.cuda()
net.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

eval_loss = np.empty((dataset.get_eval_length(),))

total_iterations = 0
for epoch in range(n_epoch):
    '''
    TRAIN
    '''
    for batch, labels in dataloader:
        batch=batch.cuda()
        labels=labels.cuda()

        optimizer.zero_grad()
        output = net(batch)

        loss = criterion(output, labels)
        loss.backward()

    '''
    VALIDATION
    '''
    if total_iterations % 500 == 0:
        net.eval()

        with torch.no_grad():
            for j in range(dataset.get_eval_length()):
                image, label = dataset.get_eval_item(j)
                image = image.cuda()
                label = label.cuda()

                output = net(batch)

                eval_loss[j] = criterion(output, label)

            # save best chkpt
            if np.mean(eval_loss) < best_eval_loss:
                fckpt_name = '{}/MODEL1_BEST.pth'.format(
                    chkpt_dir)
                torch.save(net.state_dict(), fckpt_name)
                best_eval_loss = np.mean(eval_loss)

        net.train()
            
    optimizer.step()