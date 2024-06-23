# Image Caption Generator with CNN & LSTM

This project aims to generate captions for images using a Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory (LSTM) network for sequence modeling. The project utilizes the Flickr8k dataset for training and evaluation.

![Finished App](https://github.com/HGSChandeepa/Image-Caption-Generator/blob/main/core.png)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Generating Captions](#generating-captions)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Introduction

Image captioning involves generating textual descriptions for given images. This project combines the power of CNNs and LSTMs to create a model that can produce meaningful captions for images.

## Dataset

The dataset used in this project is the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k/data). It consists of 8,000 images each paired with five different captions.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/HGSChandeepa/Image-Caption-Generator.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Image-Caption-Generator
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

### Extract Features from Images

This block of code is responsible for extracting features from images using a pre-trained model. It loads each image from a specified directory, preprocesses the image to the required input format, and uses the pre-trained model to generate feature vectors for each image. These features are then stored in a dictionary with the image ID as the key.

### Store Features in Pickle

This block of code saves the extracted image features to a pickle file. The features dictionary, which contains the feature vectors for each image, is serialized and stored in a file named 'features.pkl' in the specified working directory. This allows for easy loading and reuse of the preprocessed features in future stages of the project.

### Tokenize the Text

This block of code tokenizes the text captions. It creates a `Tokenizer` object from the Keras library, which is used to vectorize a text corpus by turning each text into either a sequence of integers or a vector. The tokenizer is then fitted on all the captions, which builds the word index based on the frequency of words in the captions. The vocabulary size is determined by the number of unique tokens found, which is the length of the tokenizer's word index plus one (to account for the reserved index for padding).

### Split Data into Training and Testing Sets

This block of code splits the dataset into training and testing sets. It first creates a list of image IDs from the keys of the `mapping` dictionary, which presumably maps images to their corresponding captions. It then calculates a split index to divide the dataset, typically set to 90% for training and 10% for testing. The list of image IDs is sliced into two separate lists: `train` for training data and `test` for testing data.

## Model Architecture

The model combines a CNN for image feature extraction and an LSTM for generating captions. 

## Training the Model

Train the model using the training dataset. Specify the loss function, optimizer, and run the training loop.

## Generating Captions

Use the trained model to generate captions for new images.

## Evaluation

Evaluate the model's performance using metrics like BLEU score and visualize the results with example images and their predicted captions.

## Results

Showcase some example images along with their generated captions and the BLEU scores.

## Conclusion

Summarize the project, discuss the results, and suggest potential improvements or future work.

## Acknowledgements

- The dataset used in this project is from Kaggle: [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k/data).
- This project was created with the help of various open-source libraries and pre-trained models.
