import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from dataloader import trainloader, testloader
from models import Resnet_32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

resnet18 = Resnet_32.resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
torch.save(resnet18, './checkpoints/resnet18.pth')

print('Finished Training')