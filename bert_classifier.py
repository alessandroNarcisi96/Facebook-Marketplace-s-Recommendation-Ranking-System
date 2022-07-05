import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import warnings
from tqdm import tqdm

class TextClassifier(nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768):
        super(TextClassifier, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(192 , 128))

    def forward(self, inp):
        x = self.main(inp)
        return x