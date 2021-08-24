from torch.utils.data import Dataset, DataLoader
from skimage import io
from glob import glob
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np


transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    #  transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 32

class MyCifarSet(Dataset):
    # data_path_list - 이미지 path 전체 리스트
    # label - 이미지 ground truth
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = self.get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            # 뒤에서 두번째가 class다.
            label_list.append(path.split('\\')[-2])
        return label_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


DATA_PATH_TRAINING_LIST = glob('./datasets/CIFAR10_64_bicubic/train/*/*.jpg')
DATA_PATH_TESTING_LIST = glob('./datasets/CIFAR10_64_bicubic/test/*/*.jpg')

trainloader = torch.utils.data.DataLoader(
    MyCifarSet(
        DATA_PATH_TRAINING_LIST,
        classes,
        transform=transform_train
    ),
    batch_size=batch_size,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    MyCifarSet(
        DATA_PATH_TESTING_LIST,
        classes,
        transform=transform_test
    ),
    batch_size=batch_size,
    shuffle=False
)

if __name__=='__main__':
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))