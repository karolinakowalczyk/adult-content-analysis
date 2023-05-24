import os
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def getModel():
    image_model = EfficientNet.from_name('efficientnet-b4', num_classes=1000)
    image_model = nn.Sequential(image_model,
                      nn.Dropout(p=0.2, inplace=True),
                      nn.Linear(1000, 512),
                      nn.BatchNorm1d(512),
                      MemoryEfficientSwish(),
                      nn.Linear(512, 1),
                      nn.Sigmoid())
    image_model.load_state_dict(torch.load(os.getcwd() + '\model.pth'))
    return image_model