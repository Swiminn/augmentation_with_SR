import os
from transform_image import bicubic, liif

print(os.listdir('datasets/CIFAR10/test'))
bicubic_64_dir = 'datasets/CIFAR10_64_bicubic'
liif_64_dir = 'datasets/CIFAR10_64_liif'
cifar10_dir = 'datasets/CIFAR10'

for dataset in os.listdir(cifar10_dir):
    dataset_dir = os.path.join(cifar10_dir, dataset)
    for category in os.listdir(dataset_dir):
        category_dir = os.path.join(dataset_dir, category)
        if os.path.isdir(os.path.join(bicubic_64_dir, dataset, category)) is False:
            os.makedirs(os.path.join(bicubic_64_dir, dataset, category))
        if os.path.isdir(os.path.join(liif_64_dir, dataset, category)) is False:
            os.makedirs(os.path.join(liif_64_dir, dataset, category))
        print(category_dir)
        for i, picture in enumerate(os.listdir(category_dir)):
            img = liif(os.path.join(category_dir, picture), 64, 64)
            img.save(os.path.join(liif_64_dir, dataset, category, picture))
            if i % 100 == 0:
                print("processing: ", i)