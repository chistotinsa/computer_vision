# computer_vision

This repository contains neural networks for various computer vision tasks.

1. Image predictor with CLI 
This CNN model is written to predict the image class (the layers dementions are sharpened for the MNIST, but it`s easy to recode it to some different images so far as model`s power allows to do it). It has a command line interface (CLI) function that allows you to pass the data by mentioning CSV with [image-paths,labels] and setting output file path in command line by using flags.

2. COVID_19_detection_CNN
This model can detect COVID-19 using x-ray images of lungs with accuracy ~1.0. The model is trained on a dataset of x-ray images from Kaggle
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
with the goal of identifying COVID-19 cases based on lung x-ray images.

Usage
Clone the repository and install the required dependencies.
