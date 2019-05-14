import torch
from torch import nn

class FASTCNN(nn.Module):
    def __init__(self):
        super(FASTCNN,self).__init__()
        print('I am the FASTCNN MODEL')