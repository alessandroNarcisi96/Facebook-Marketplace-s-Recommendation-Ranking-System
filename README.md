# Facebook-Marketplace
## Introduction

### Learning Goal
The goal of this project is simulating how Facebook markeplace categorizes the products automatically.
The dataset basically contains images and each of them has a description.
The target variable the predict is the category.

The idea is building a model that is able to analyze both images and text at the same time.

### Images
Needless to say to understand complex images we need to use CNN and in this case I will use Pytorch.
I will use a tecnique very powerfull called Transfer Learning.
It consists in using a pre-trained model and fine it by adding a final linear layer.
So in this case I will use a pre-trained model called nvidia_resnet50.

### Text
To perform the word-embedding I used the BERT(Bidirectional Encoder Representations from Transformers).
It turns out that the performance of this model are really good as it connects a word to its context in order to understand better the meaning.
So in this case I will apply a transfer learning and I will tune just the final layer with the descriptions of the products provided by the dataset.


## Milestone 1 - Data Cleaning
The images have different formats so the strategy is to crop them and paste them on a black background.<br>
This operation is performed in clean_images.py.<br>

The descriptions are full of emoticons, punctuation and number so we have to clean the words as well.<br>
Then we have to encode the categories in numbers.<br>

## Milestone 2 - Create the DataSet and the DataLoader
As I am using Pytorch I need to create a dataset.<br>
The items contained are basically a triple formed by three elements: an image, a description and the category.<br>

ImageTextDataset inherites from torch.utils.data.Dataset and has two main methods:<br>
__init__ and __getitem__.<br>
In the first one what I do is initialize the BertModel and take the useful data from the main dataset.<br>
In __getitem__ the image is loaded and transformed in a tensor.The description is tokenized and then I perform the embedding by the bert model.<br>
Lastly, I create the triple formed by the image,the embedded description and the encoded category.<br>

## Milestone 3 - Create the CNN to analyze text
The CNN to analyze text is designed as a neural network with 4 convolution layers.<br>
To see how a CNN works with text check out here: https://lena-voita.github.io/nlp_course/models/convolutional.html

## Milestone 4 - Combine the image model and the text model in one
As visible in combined_model.py Pytorch offers a way very easy to combine two models in one by using torch.cat().<br>
On the top of the neural netowrk there is a liner layer.


## Milestone 5 - Performance

At the end of project.ipynb there is the training and the validation.
The lost function is the CrossEntropy with 5 epochs.

The result with 13 categories is quite good:90% accuracy on training set and 85% accuracy on validation set
