import torch.nn as nn
import torch
def mse(input, target):
    with torch.no_grad():
        loss = nn.MSELoss()
        output = loss(input, target)
    return output