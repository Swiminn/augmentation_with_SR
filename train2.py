import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from dataloader import trainloader, testloader
from models import Resnet_32, Residual2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

resnet18 = Residual2.resnet18(10, 3).to(device)
print(resnet18)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)


def adjust_learning_rate(optimizer, lr):
    lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    running_corrects = 0
    if epoch > 0 and (epoch + 1) % 35 == 0 :
        adjust_learning_rate(optimizer, learning_rate)
        learning_rate = learning_rate / 10

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predictions = torch.max(outputs, 1)
        # print statistics
        running_loss += loss.item()
        running_corrects += torch.sum(predictions == labels).item()

    epoch_loss = running_loss / (len(trainloader))
    epoch_acc = running_corrects / (32 * len(trainloader))
    print('epoch : [%d] loss: %.4f' %(epoch + 1, epoch_loss))
    print('epoch_correct : ', epoch_acc)
torch.save(resnet18, './checkpoints/resnet18.pth')

print('Finished Training')