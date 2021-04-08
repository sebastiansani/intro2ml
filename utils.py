import torch
import network
from prettytable import PrettyTable
import inspect


def get_total_grad(net):
    grads = []
    for param in net.parameters():
        if param.requires_grad:
            grads.append(param.grad.view(-1))
    return torch.cat(grads).sum()


def count_parameters(model, name):
    print(name)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    print(table)
    return total_params


def get_network_class_names():
    net_classes = inspect.getmembers(network, inspect.isclass)
    net_class_names = [x[0] for x in net_classes]
    return net_class_names
