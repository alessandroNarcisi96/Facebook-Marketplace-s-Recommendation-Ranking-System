import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle
import bert_classifier


class CombinedModel(nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768,
                 num_classes: int = 2):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = resnet50.fc.out_features
        self.image_classifier = nn.Sequential(resnet50, nn.Linear(out_features, 128)).to(device)
        self.text_classifier = bert_classifier.TextClassifier(ngpu=ngpu, input_size=input_size)
        self.main = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features