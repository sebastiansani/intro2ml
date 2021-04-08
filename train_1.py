import torch
from dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import vgg_pretrained

chkpt_dir = 'chkpts'
dataset_root = 'dataset'
n_epoch = 50
best_eval_loss = 9999
batch_size = 32

class_sample_count = [1840, 757, 1311, 1297, 996, 1280, 1851, 1419, 1430, 234]
weights = 1 / torch.Tensor(class_sample_count)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
weights = weights.double()

dataset = Dataset(dataset_root, set_seed=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=sampler)

net = vgg_pretrained()

net.cuda()
net.train()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

eval_loss = np.empty((dataset.get_eval_length(),))

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
        for j in range(dataset.get_eval_length()):
            image, label = dataset.get_eval_item(j)
            image = image.cuda()
            label = label.cuda()

            image = image.unsqueeze(0)
            label = label.unsqueeze(0)

            output = net(image)

            eval_loss[j] = criterion(output, label)

        print('e{} - train loss {:.4f}, valid loss {:.4f}'.format(
            epoch, loss.item(), np.mean(eval_loss)))

        # save best chkpt
        if np.mean(eval_loss) < best_eval_loss:
            fckpt_name = '{}/MODEL1_BEST.pth'.format(
                chkpt_dir)
            torch.save(net.state_dict(), fckpt_name)
            best_eval_loss = np.mean(eval_loss)

    net.train()

    scheduler.step()
