import numpy as np
import tensorboard_manager
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

def create_words(columnText):
    words = set()
    for x in columnText:
        words = words.union(set(x.split(" ")))
    return list(words)

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

def fillzero(list,max):
    to_add = max - len(list)
    zeros_list = [0 for i in range(to_add)]
    list.extend(zeros_list)
    return list


def to_tensor(list):
    return torch.LongTensor(list)

class ProductTextDataset(Dataset):
    def __init__(self,vectors,ds):
        super().__init__()
        categories= []
        for n, item in enumerate(vectors):
            cat = ds.iloc[[n]].category
            categories.append(cat)
           
        self.X = to_tensor(vectors)
        self.Y = categories
        assert len(self.X) == len(self.Y)

    def __getitem__(self,index):
        features = self.X[index]
        label = self.Y[index]
        return features,label
        
    def __len__(self):
        return len(self.X)

class NNLM(torch.nn.Module):
    def __init__(self,n_class,n_step,n_hidden,m):
       self.n_step = n_step
       self.n_hidden = n_hidden 
       self.m = m 
       super(NNLM, self).__init__()
       self.embeddings = torch.nn.Embedding(n_class, self.m) #Embedding layer:first parameter is how many words we have and the second is the dimension of each vector
       self.hidden1 = torch.nn.Linear(n_step * self.m, self.n_hidden, bias=False)
       self.ones = torch.nn.Parameter(torch.ones(self.n_hidden))
      
       self.hidden2 = torch.nn.Linear(self.n_hidden, n_class, bias=False)
       self.hidden3 = torch.nn.Linear(n_step * self.m, n_class, bias=False) #final layer
      
       self.bias = torch.nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        word_embeds = self.embeddings(X) # embeddings
        print(word_embeds.shape)
        X = word_embeds.view(-1, self.n_step * self.m) # first layer
        tanh = torch.tanh(self.ones + self.hidden1(X)) # tanh layer
        output = self.bias + self.hidden3(X) + self.hidden2(tanh) # summing up all the layers with bias
        return word_embeds, output
    
def make_batch(sentences,word2id):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() 
        input = [word2id[n] for n in word[:-1]]
        target = word2id[word[-1]] 

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch