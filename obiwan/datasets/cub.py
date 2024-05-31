# Credit: https://github.com/ejkim47/prob-cbm and https://github.com/yewsiang/ConceptBottleneck
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torch
import os
from pathlib import Path


def cub_classification_data(pkl_dir: str):
    train = pickle.load(open(pkl_dir + '/train.pkl', 'rb'))
    test = pickle.load(open(pkl_dir + '/test.pkl', 'rb'))
    val = pickle.load(open(pkl_dir + '/val.pkl', 'rb'))

    resol = 224
    resized_resol = int(299 * 256 / 224)
    
    train_transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            # transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    
    test_transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    
    train_data = []
    test_data = []
    val_data = []

    for idx, img_data in enumerate(train):
        img_path = img_data['img_path']

        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        train_data.append((img, attr_label, class_label))

    for idx, img_data in enumerate(test):
        img_path = img_data['img_path']
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        test_data.append((img, attr_label, class_label))

    for idx, img_data in enumerate(val):
        img_path = img_data['img_path']
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'val'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        val_data.append((img, attr_label, class_label))

    train_imgs, train_attrs, train_labels = zip(*train_data)
    test_imgs, test_attrs, test_labels = zip(*test_data)
    val_imgs, val_attrs, val_labels = zip(*val_data)

    train_imgs = [train_transform(img) for img in train_imgs]
    train_attrs = [torch.Tensor(attr) for attr in train_attrs]
    train_labels = [torch.Tensor([label]) for label in train_labels]
    train_imgs = torch.stack(train_imgs)
    train_attrs = torch.stack(train_attrs)
    train_labels = torch.stack(train_labels)

    test_imgs = [test_transform(img) for img in test_imgs]
    test_attrs = [torch.Tensor(attr) for attr in test_attrs]
    test_labels = [torch.Tensor([label]) for label in test_labels]
    test_imgs = torch.stack(test_imgs)
    test_attrs = torch.stack(test_attrs)
    test_labels = torch.stack(test_labels)

    val_imgs = [test_transform(img) for img in val_imgs]
    val_attrs = [torch.Tensor(attr) for attr in val_attrs]
    val_labels = [torch.Tensor([label]) for label in val_labels]
    val_imgs = torch.stack(val_imgs)
    val_attrs = torch.stack(val_attrs)
    val_labels = torch.stack(val_labels)

    return train_imgs, train_attrs, train_labels, test_imgs, test_attrs, test_labels, val_imgs, val_attrs, val_labels

def get_cub_classification_dataloaders(pkl_dir: str, batch_size: int, num_workers: int):
    train_imgs, train_attrs, train_labels, test_imgs, test_attrs, test_labels, val_imgs, val_attrs, val_labels = cub_classification_data(pkl_dir)
    train_loader = torch.utils.data.DataLoader(TensorDataset(train_imgs, train_attrs, train_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_imgs, test_attrs, test_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(TensorDataset(val_imgs, val_attrs, val_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader, val_loader


def get_cub_data(pkl_dir: str):
    train = pickle.load(open(pkl_dir + '/train.pkl', 'rb'))
    test = pickle.load(open(pkl_dir + '/test.pkl', 'rb'))
    val = pickle.load(open(pkl_dir + '/val.pkl', 'rb'))

    resol = 224
    resized_resol = int(299 * 256 / 224)
    
    train_transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            # transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    
    test_transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])

    class_to_data_map = {}

    for idx, img_data in enumerate(train):
        img_path = img_data['img_path']

        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        if class_label not in class_to_data_map:
            class_to_data_map[class_label] = []
        class_to_data_map[class_label].append((img, attr_label, class_label))

    for idx, img_data in enumerate(test):
        img_path = img_data['img_path']
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        if class_label not in class_to_data_map:
            class_to_data_map[class_label] = []
        class_to_data_map[class_label].append((img, attr_label, class_label))

    for idx, img_data in enumerate(val):
        img_path = img_data['img_path']
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[idx:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'val'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = img_path.replace('CUB_200_2011', 'CUB200')
            img_path = os.path.join(Path(pkl_dir).parent.parent, img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        attr_label = img_data['attribute_label']

        if class_label not in class_to_data_map:
            class_to_data_map[class_label] = []
        class_to_data_map[class_label].append((img, attr_label, class_label))

    # data with class less than 100 in train
    
    train_data = []
    test_data = []

    for class_label in class_to_data_map:
        if class_label < 100:
            train_data.extend(class_to_data_map[class_label])
        else:
            test_data.extend(class_to_data_map[class_label])

    train_imgs, train_attrs, train_labels = zip(*train_data)
    test_imgs, test_attrs, test_labels = zip(*test_data)
    train_imgs = [train_transform(img) for img in train_imgs]
    train_attrs = [torch.Tensor(attr) for attr in train_attrs]
    train_labels = [torch.Tensor([label]) for label in train_labels]
    train_imgs = torch.stack(train_imgs)
    train_attrs = torch.stack(train_attrs)
    train_labels = torch.stack(train_labels)
    train_data = TensorDataset(train_imgs, train_attrs, train_labels)

    test_imgs = [test_transform(img) for img in test_imgs]
    test_attrs = [torch.Tensor(attr) for attr in test_attrs]
    test_labels = [torch.Tensor([label]) for label in test_labels]
    test_imgs = torch.stack(test_imgs)
    test_attrs = torch.stack(test_attrs)
    test_labels = torch.stack(test_labels)
    test_data = TensorDataset(test_imgs, test_attrs, test_labels)

    return train_data, test_data


def get_cub_dataloaders(pkl_dir: str, batch_size: int, num_workers: int):
    train_data, test_data = get_cub_data(pkl_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

