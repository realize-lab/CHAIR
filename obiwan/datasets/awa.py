# Credit: https://github.com/dfan/awa2-zero-shot-learning/blob/master/AnimalDataset.py

import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


SELECTED_CONCEPTS_STR = ["BLACK", "WHITE", "BLUE", "BROWN", "GRAY", "ORANGE", "RED", "YELLOW", "PATCHES", "SPOTS", "STRIPES", "FURRY", "HAIRLESS", "TOUGHSKIN", "BIG", "SMALL", "BULBOUS", "LEAN", "FLIPPERS", "HANDS", "HOOVES", "PADS", "PAWS", "LONG LEG", "LONG NECK", "TAILS", "HORNS", "CLAWS", "TUSKS", "BIPEDAL", "QUADRAPEDAL", "ARCTIC", "COASTAL", "DESERT", "BUSH", "PLAINS", "FOREST", "FIELDS", "JUNGLE", "MOUNTAINS", "OCEAN", "GROUND", "WATER", "TREE", "CAVE"]

SELECTED_CONCEPTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 44, 45, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]

assert os.environ.get('AWA_PATH') is not None, "Please set the environment variable AWA_PATH to the path of the Animals with Attributes dataset"

AWA_PATH = os.environ.get('AWA_PATH')

class AnimalDataset(data.dataset.Dataset):
  def __init__(self, classes_file, transform):
    predicate_binary_mat = np.array(np.genfromtxt(f'{AWA_PATH}predicate-matrix-binary.txt', dtype='int'))
    self.predicate_binary_mat = predicate_binary_mat
    self.transform = transform

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(f'{AWA_PATH}classes.txt') as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    img_names = []
    img_index = []
    with open(f'{AWA_PATH}'.format(classes_file)) as f:
      for line in f:
        class_name = line.strip()
        FOLDER_DIR = os.path.join('{AWA_PATH}JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor)

        class_index = class_to_index[class_name]
        for file_name in files:
          img_names.append(file_name)
          img_index.append(class_index)
    self.img_names = img_names
    self.img_index = img_index

    print("Num classes: ", len(class_to_index))

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)
    if im.shape != (3,224,224):
      print(self.img_names[index])

    im_index = self.img_index[index]
    im_predicate = self.predicate_binary_mat[im_index, SELECTED_CONCEPTS]
    return im, im_predicate, im_index

  def __len__(self):
    return len(self.img_names)

def get_awa_dataloaders(batch_size, num_workers):
  train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((224,224)), 
    transforms.ToTensor()
  ])

  test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
  ])

  train_dataset = AnimalDataset('trainclasses.txt', train_transform)
  test_dataset = AnimalDataset('testclasses.txt', test_transform)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

  return train_loader, test_loader

def get_awa_classification_loaders(batch_size, num_workers):
  train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((224,224)), 
    transforms.ToTensor()
  ])

  test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
  ])

  train_dataset = AnimalDataset('trainclasses.txt', train_transform)


  # split train dataset into train and validation
  train_dataset, test_dataset = data.random_split(train_dataset, [0.8, 0.2])

  # set test dataset transform to test_transform
  test_dataset.transform = test_transform

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

  return train_loader, test_loader

