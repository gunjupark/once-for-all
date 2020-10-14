import torch
import torch.nn as nn
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split


# Helper for constructing the one-hot vectors.


# Tabular Dataset
class BPNetDataset(Dataset):
    #Args. csv_File(str) : Path to the csv file contain tabular data.

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        
        #self.data = pd.get_dummies(data_frame)
        

        ''' 
        #Naive Method
        self.X = self.data.drop(["Normalized Time","Normalized Accuracy"],axis=1).copy().values.astype(np.float32)
        self.y = self.data["Normalized Time"].copy().values.astype(np.float32)
        '''
        target_time = "Normalized Time"
        target_acc = "Normalized Accuracy"

        thres = ["r0","r1","r2","r3","r4","r5","r6","r7","r8"]
        auxs = ["act0","act1","act2","act3","act4","act5","act6","act7","act8"]
        

        self.x_aux = self.data.drop([target_time,target_acc]+thres,axis=1).copy().values.astype(np.float32)
        self.x_thres = self.data.drop([target_time,target_acc]+auxs,axis=1).copy().values.astype(np.float32)
        self.y = self.data["Normalized Time"].copy().values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.x_aux[idx], self.x_thres[idx], self.y[idx]]


class MLP_Predictor(nn.Module):

    def __init__(self,input_size, output_size):

        super().__init__()

        self.fc1 = nn.Linear(input_size, 700)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(700, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 300)
        self.relu3 = nn.ReLU()
        #self.fc4 = nn.Linear(300, 300)
        #self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(300, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        #x = self.fc4(x)
        #x = self.relu4(x)
        x = self.fc5(x)

        return x


class BP_Predictor(MLP_Predictor):
    def __init__(self, aux_num, output_size):
        super().__init__(aux_num*2 ,output_size)

        self.bn_thres = nn.BatchNorm1d(aux_num)
        self.drop = nn.Dropout(0.3)
        self.bn700 = nn.BatchNorm1d(700)
        self.bn500 = nn.BatchNorm1d(500)
        self.bn300 = nn.BatchNorm1d(300)

    def forward(self, aux, thres):
        x1 = aux
        x2 = self.bn_thres(thres)
        x = torch.cat([x1,x2], 1)
        x = self.fc1(x)
        x = self.bn700(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn500(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn300(x)
        x = self.relu3(x)
        x = self.drop(x)
        #x = self.fc4(x)
        #x = self.bn300(x)
        #x = self.relu4(x)
        x = self.fc5(x)

        return x

def train(csv_file, n_epochs=10000, split_rate=0.2):

    #Load Dataset
    dataset = BPNetDataset(csv_file)

    train_size = int(split_rate * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=train_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_size, shuffle=True)


    device = torch.device("cuda:0")

    #first_net = MLP_Predictor(18,1)
    first_net = BP_Predictor(9,1)

    #net = first_net.to(device)
    bpnet = first_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(bpnet.parameters(), weight_decay=0.0001, lr=0.001)

    loss_per_iter, loss_per_batch = [], []


    for epoch in range(n_epochs):

        train_loss = 0.0

        for i, (auxs, thres, labels) in enumerate(trainloader):
            auxs = auxs.to(device)
            thres = thres.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()


            #Forward & Backward & Optimize

            #outputs = net(inputs)
            outputs = bpnet(auxs, thres)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loss_per_iter.append(loss.item())

        loss_per_batch.append(train_loss / (i+1))



    # Comparing training to test
    '''
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs.float())
    '''
    dataiter = iter(testloader)
    auxs, thres, labels = dataiter.next()
    auxs = auxs.to(device)
    thres = thres.to(device)
    labels = labels.to(device)
    outputs = bpnet(auxs, thres)
    print("Root mean squared error")
    print("Training:", np.sqrt(loss_per_batch[-1]))
    print("Test", np.sqrt(criterion(labels.float(), outputs).detach().cpu().numpy()))
    '''
    # Plot training loss curve
    plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per epoch")
    plt.plot(np.arange(len(loss_per_iter), step=4) + 3, loss_per_batch, ".-", label="Loss per mini-batch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    import os
    import sys
    import argparse

    csv_file = os.path.join(sys.path[0], "test_data.csv")

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", "-c", type=str, nargs="?", help="Dataset files")
    parser.add_argument("--epochs", "-e",type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--rate", "-r",type=float, default=0.2, help="Split rate for trainset , testset split")

    args= parser.parse_args()


    train(args.csv, args.epochs, args.rate)


