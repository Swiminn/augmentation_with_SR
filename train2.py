import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from dataloader import trainloader, testloader, batch_size
from models import Resnet_32, Residual2
from models.resnet_paper import resnet20
from models.resnet_95 import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# resnet18 = Residual2.resnet18(10, 3).to(device)
resnet20 = ResNet18().to(device)
# resnet20 = Resnet_32.resnet18().to(device)
print(resnet20)


criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = optim.SGD(resnet20.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])


for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    running_corrects = 0
    resnet20.train()

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet20(inputs)
        # outputs = outputs.view(outputs.size(0), -1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predictions = torch.max(outputs, 1)
        # print statistics
        running_loss += loss.item()
        running_corrects += torch.sum(predictions == labels).item()
    scheduler.step()

    epoch_loss = running_loss / (len(trainloader))
    epoch_acc = running_corrects / (batch_size * len(trainloader))

    resnet20.eval()
    with torch.no_grad():
        running_test_loss = 0.0
        running_test_corrects = 0
        for i, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            test_outputs = resnet20(test_images)
            # test_outputs = test_outputs.view(test_outputs.size(0), -1)
            _, test_predictions = torch.max(test_outputs, 1)
            test_loss = criterion(test_outputs, test_labels)
            running_test_loss += test_loss.item()
            running_test_corrects += torch.sum(test_predictions == test_labels).item()
        running_test_loss = running_test_loss / (len(testloader))
        running_test_corrects = running_test_corrects / (batch_size * len(testloader))

    print('epoch : [%d] train_loss: %.4f' % (epoch + 1, epoch_loss), end=", ")
    print("test_loss : %.4f" % running_test_loss, end=", ")
    print('train_correct : %.4f' % epoch_acc, end=", ")
    print("test_correct : %.4f" % running_test_corrects)


torch.save(resnet20, './checkpoints/resnet18.pth')

print('Finished Training')