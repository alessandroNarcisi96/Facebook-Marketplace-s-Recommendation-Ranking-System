import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import requests
import random
import torch
import os
import random
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


def repeat_channel(x):
            return x.repeat(3, 1, 1)

class ImageTextDataset(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model
    Parameters
    ----------
    root_dir: str
        The directory with subfolders containing the images
    transform: torchvision.transforms 
        The transformation or list of transformations to be 
        done to the image. If no transform is passed,
        the class will do a generic transformation to
        resize, convert it to a tensor, and normalize the numbers
    
    Attributes
    ----------
    files: list
        List with the directory of all images
    labels: set
        Contains the label of each sample
    encoder: dict
        Dictionary to translate the label to a 
        numeric value
    decoder: dict
        Dictionary to translate the numeric value
        to a label
    '''

    def __init__(self,
                 ds,
                 transform: transforms = None,
                 max_length: int = 50
                ):
        super().__init__()  
        if not os.path.exists('data/resized_images'):
            raise RuntimeError('Image Dataset not found, use download=True to download it')
        self.products = ds

        self.descriptions = self.products['product_description'].to_list()
        self.labels = self.products['category'].to_list()
        self.max_length = max_length
        # Get the images        
        self.files = self.products['image_id']
        self.num_classes =13
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        label = self.labels[index]
        #print("la label e "+str(label))
        label = torch.as_tensor(label)
        image = Image.open('data/resized_images/' + self.files[index] + '.jpg')
        if image.mode != 'RGB':
          image = self.transform_Gray(image)
        else:
          image = self.transform(image)

        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)

        return image, description, label

    def __len__(self):
        return len(self.files)



def split_train_test(dataset):
    train_dataset = dataset.head(50)  
    validation_dataset = dataset.tail(50)
    validation_dataset = validation_dataset.reset_index()
    return train_dataset, validation_dataset

