import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

import torch.optim as optim

def computeAcc(testset, net):

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = net(X.view(-1,784))
            #print(output)
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    return correct/total


def evaluateModel (modelDf):
  op = modelDf.index.get_level_values('optimizerType')[0]
  lr = modelDf.index.get_level_values('learningRate')[0]
  epochnum = modelDf.index.get_level_values('epoch')[0]
  batch = int(modelDf.index.get_level_values('batchSize')[0])
  
  trainset = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True)
  testset = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False)

  net = Net()

  if op == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr)
  else:
    optimizer = optim.SGD(net.parameters(), lr=lr)

  EPOCHS = epochnum
  #print(EPOCHS)
  for epoch in range(EPOCHS): # 3 full passes over the data
    for data in trainset:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1,28*28))  # pass in the reshaped batch (recall they are 28x28 atm)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    #print(epoch)

  accuracy = computeAcc(testset, net)
  resultSe = pd.Series({'accuracy': accuracy})
  # print(resultSe['accuracy'])
  return resultSe


def drawLinePlot (plotDf, ax):
  for batchSize, subDf in plotDf.groupby('batchSize'):
    subDf = subDf.droplevel('batchSize')
    subDf.plot.line(ax = ax, label = 'batchSize = {}'.format(batchSize), y = 'accuracy', marker = 'o')
  ax.set_xlabel('epoch')


def main():
  optimizerType = ["Adam", "SGD"]
  learningRate = [0.00001, 0.001, 0.1]
  epoch = list(range(1,10+1))
  batchSize = [32, 64, 128, 256]

  levelValues = [optimizerType, learningRate, epoch, batchSize]
  levelNames = ["optimizerType", "learningRate", "epoch", "batchSize"]

  modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

  toSplitFrame = pd.DataFrame(index = modelIndex)

  modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateModel)


  fig = plt.figure(figsize=(15, 15))
  plotLevels = ['epoch', 'batchSize']
  plotRowNum = len(learningRate)
  plotColNum = len(optimizerType)
  plotCounter = 1

  for (key, plotDf) in modelResultDf.groupby(['learningRate', 'optimizerType']):
    plotDf.index = plotDf.index.droplevel(['learningRate', 'optimizerType'])
    ax = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
    drawLinePlot(plotDf, ax)
    plotCounter += 1
  
  fig.text(x=0.5, y=0.04, s='optimizerType', ha='center', va='center')
  fig.text(x=0.05, y=0.5, s='learningRate', ha='center', va='center', rotation=90)

  plt.show()


if __name__ == "__main__":
    main()