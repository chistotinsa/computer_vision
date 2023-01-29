#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision import transforms

def train(dataset_path, model_path):

    class TrainDataset(Dataset):
        
        def __init__(self, annotations_file, transform=Resize([28, 28]), target_transform=None):
            self.img_labels = pd.read_csv(annotations_file, encoding='utf-8')
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = self.img_labels.iloc[idx, 0]
            image = read_image(img_path)
            image = image.type(torch.float32) 
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

    train_dataset = TrainDataset(dataset_path)

    batch_size = 32

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Flattener(nn.Module):
        def forward(self, x):
            batch_size, *_ = x.shape
            return x.view(batch_size, -1)

    nn_model = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                nn.BatchNorm2d(6),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2, stride = 2),
                                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2, stride = 2),
                                Flattener(),
                                nn.Linear(256, 300),
                                nn.BatchNorm1d(300),
                                nn.ReLU(),
                                nn.Linear(300, 120),
                                nn.BatchNorm1d(120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.BatchNorm1d(84),
                                nn.ReLU(),
                                nn.Linear(84, 10),
                                nn.LogSoftmax(dim=-1))

    nn_model.type(torch.FloatTensor)
    nn_model.to(device)

    loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
    optimizer = optim.SGD(nn_model.parameters(), lr=0.1, weight_decay=0.0001)
    
    EPOCHS = 6

    def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs): 
        
        loss_history = []
        train_history = []
        val_history = []
        for epoch in range(num_epochs):
            model.train() # Enter train mode

            loss_accum = 0
            correct_samples = 0
            total_samples = 0
            for i_step, (x, y) in enumerate(train_loader):

                x_gpu = x.to(device)
                y_gpu = y.to(device)
                prediction = model(x_gpu)    
                loss_value = loss(prediction, y_gpu)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                _, indices = torch.max(prediction, 1)
                correct_samples += torch.sum(indices == y_gpu)
                total_samples += y.shape[0]

                loss_accum += loss_value

            ave_loss = loss_accum / i_step
            train_accuracy = float(correct_samples) / total_samples
            val_accuracy = compute_accuracy(model, val_loader)

            loss_history.append(float(ave_loss))
            train_history.append(train_accuracy)
            val_history.append(val_accuracy)

            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

        return loss_history, train_history, val_history

    def compute_accuracy(model, loader):
        """
        Computes accuracy on the dataset wrapped in a loader
        Returns: accuracy as a float value between 0 and 1
        """

        model.eval() # Evaluation mode

        for _, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss_val = loss(prediction, y)

            accuracy = (torch.argmax(prediction, dim=1) == y).sum().item() / prediction.shape[0]

        return accuracy

    loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, EPOCHS)

    path = model_path
    PATH = os.path.join(path, "MNIST_predictor.pt")
    
    torch.save({
            'epoch': EPOCHS,
            'model_state_dict': nn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, PATH)
    
    pass

def inference(model_path, input_path, output_path):

    pass



def main():
    
    parser = argparse.ArgumentParser(description=r"Модель, предсказывающая значение цифры на MNIST с некоторым функционалом")
    
    parser.add_argument("--dataset", type=str, required=True, help=r"Укажите путь к CSV файлу с датасетом для тренировки модели, например --dataset 'C:\Users\user\Desktop\file.csv'")
    parser.add_argument("--model", type=str, required=True, help=r"Укажите путь для сохранения модели, например --model 'C:\Users\user\Desktop\'")

    args = parser.parse_args()


    if args.dataset is None or args.model is None:
        parser.error(r"Необходимо указать путь к датасету и путь к файлу для сохранения модели, для подсказки введите help")

    train(args.dataset, args.model)


if __name__ == "__main__":
    main()


