
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# fix random seed
torch.manual_seed(1580)

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))


trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)


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

net = Net()


optimizer = optim.Adam(net.parameters(), lr=0.001)   # first arg is what parameters are adjustable  1e-3

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
    return round(correct/total, 3)

model_one_acc = []

for epoch in range(10): # 3 full passes over the data
    for data in trainset:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1,28*28))  # pass in the reshaped batch (recall they are 28x28 atm) (-1 means any dim)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    model_one_acc.append(computeAcc(testset, net))
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 


fig = plt.figure()
epoch = range(1, 11)
plt.plot(epoch, model_one_acc, 'o-', label='model_one')
#plt.plot(epoch, one_valid_acc, 'o-', label='one_valid')
#plt.plot(epoch, two_train_acc, 'o-', label='two_train')
#plt.plot(epoch, two_valid_acc, 'o-', label='two_valid')
plt.xlabel(r'$epoch$')
plt.ylabel(r'$accuracy$')
plt.title('Network Test Accuracy by epoch (ADAM)')
plt.legend()
plt.show()