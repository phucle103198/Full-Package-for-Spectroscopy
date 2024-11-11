import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import torch.optim as optim
from Regression.CnnModel import ConvNet, DeepSpectra, AlexNet
import os
from datetime import datetime
from Evaluate.RgsEvaluate import ModelRgsevaluate, ModelRgsevaluatePro
import matplotlib.pyplot  as plt


LR = 0.001
BATCH_SIZE = 16
TBATCH_SIZE = 240


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)



def ZspPocessnew(X_train, X_test, y_train, y_test, need=True): 

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        #yscaler = StandardScaler()
        yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
        y_test = yscaler.transform(y_test.reshape(-1, 1))

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]

        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        data_train = MyDataset(X_train_new, y_train)
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test




def CNNTrain(NetType, X_train, X_test, y_train, y_test, EPOCH):


    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    if NetType == 'ConNet':
        model = ConvNet().to(device)
    elif NetType == 'AlexNet':
        model = AlexNet().to(device)
    elif NetType == 'DeepSpectra':
        model = DeepSpectra().to(device)



    criterion = nn.MSELoss().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # # initialize the early_stopping object
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!") 
    # to track the training loss as the model trains
    for epoch in range(EPOCH):
        train_losses = []
        model.train() 
        train_rmse = []
        train_r2 = []
        train_mae = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data 
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())
            rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
            # plotpred(pred, y_true, yscaler))
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
        avg_train_loss = np.mean(train_losses)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        print('Epoch:{}, TRAIN:rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse), (avgr2), (avgmae)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        with torch.no_grad():
            model.eval() 
            test_rmse = []
            test_r2 = []
            test_mae = []
            for i, data in enumerate(test_loader):
                inputs, labels = data 
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model(inputs)
                pred = outputs.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                rmse, R2, mae = ModelRgsevaluatePro(pred, y_true, yscaler)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)
            avgrmse = np.mean(test_rmse)
            avgr2   = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            print('EPOCH：{}, TEST: rmse:{}, R2:{}, mae:{}'.format((epoch+1), (avgrmse), (avgr2), (avgmae)))
            scheduler.step(rmse)

    return avgrmse, avgr2, avgmae











#
# def CNN(X_train, X_test, y_train, y_test, BATCH_SIZE, n_epochs):
#
#     CNNTrain(X_train, X_test, y_train, y_test,BATCH_SIZE,n_epochs)
