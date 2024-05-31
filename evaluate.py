from obiwan.new_models import CBM, FuseCBM
from obiwan.datasets.cub import get_cub_dataloaders
from obiwan.utils import recall

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MultilabelAccuracy, Accuracy, MultilabelF1Score

import os
from dotenv import load_dotenv
import uuid
import json
load_dotenv()

from typing import Tuple

import wandb #noqa

try:
    from rich.tqdm import tqdm
except ImportError:
    from tqdm import tqdm


def evaluate(model: CBM, dataloader, device, num_classes, num_concepts) -> Tuple[float, float, float]:
    """Evaluate a CBM model on the data loader for accuracy (classification setting)

    Args:
        model (CBM): CBM or FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on
        num_classes (int): Number of classes in the dataset
        num_concepts (int): Number of concepts per image in the dataset

    Returns:
        Tuple[float, float, float]: Concept accuracy, Class accuracy, Concept F1 score
    """
    model.eval()
    model.to(device)

    concept_accuracy = MultilabelAccuracy(num_labels=num_concepts)
    concept_f1 = MultilabelF1Score(num_labels=num_concepts)
    class_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    concept_accuracy.to(device)
    class_accuracy.to(device)
    concept_f1.to(device)

    with torch.no_grad():
        for imgs, attrs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            concepts, classes = model(imgs)

            concept_accuracy.update(concepts, attrs)
            class_accuracy.update(classes, labels.long().squeeze())
            concept_f1.update(concepts, attrs)

        
    final_concept_accuracy = concept_accuracy.compute()
    final_class_accuracy = class_accuracy.compute()
    final_concept_f1 = concept_f1.compute()

    return final_concept_accuracy, final_class_accuracy, final_concept_f1


def evaluate_recall(model: FuseCBM, dataloader, device, intervene: bool, pre_concept: bool):
    """Evaluate a FuseCBM model on the data loader for recall (retrieval setting)

    Args:
        model (FuseCBM): FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on
        intervene (bool): Whether to intervene or not
        pre_concept (bool): Whether to use pre-concept or not

    Returns:
        Tuple[float, float, float]: Recall@1, Recall@5, Recall@10
    """

    model.eval()
    model.to(device)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, attrs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            # embeddings = model.get_embedding(imgs)
            if pre_concept:
                embeddings = model.get_pre_concept_embedding(imgs)
            else:
                if intervene:
                    embeddings = model.get_fused_embedding(imgs, False)
                else:
                    embeddings = model.get_fused_embedding(imgs, False)
            
            embeddings = F.normalize(embeddings, dim=1)

            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    recall_list = recall(embeddings, labels, rank=[1,5,10])
    return recall_list


def evaluate_recall_with_gallery(model: FuseCBM, dataloader, device, intervene: bool, pre_concept: bool, gallery_features, gallery_labels) -> Tuple[float, float, float]:
    """Evaluate a FuseCBM model on the data loader for recall (retrieval setting) with a gallery

    Args:
        model (FuseCBM): FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on
        intervene (bool): Whether to intervene or not
        pre_concept (bool): Whether to use pre-concept embeddings or post-concept embeddings
        gallery_features (torch.Tensor): Gallery features
        gallery_labels (torch.Tensor): Gallery labels

    Returns:
        Tuple[float, float, float]: Recall@1, Recall@5, Recall@10
    """
    model.eval()
    model.to(device)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, attrs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            if pre_concept:
                embeddings = model.get_pre_concept_embedding(imgs)
            else:
                if intervene:
                    embeddings = model.get_fused_embedding_with_intervention(imgs, attrs, False, False)
                else:
                    embeddings = model.get_embedding(imgs)
            
            embeddings = F.normalize(embeddings, dim=1)

            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    recall_list = recall(embeddings, labels, rank=[1,5,10], gallery_features=gallery_features, gallery_labels=gallery_labels)
    return recall_list

