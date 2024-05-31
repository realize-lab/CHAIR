from obiwan.new_models import FuseCBM
from obiwan.utils import recall

import torch
import torch.nn.functional as F

import wandb #noqa

try:
    from rich.tqdm import tqdm
except ImportError:
    from tqdm import tqdm


def get_concepts(model, dataloader, device):
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
    model.to(device)
    model.eval()

    concepts = []
    for batch in tqdm(dataloader):
        if len(batch) == 2:
            imgs, (labels, attrs) = batch
        else:
            imgs, attrs, labels = batch
        imgs = imgs.to(device)
        attrs = attrs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            concept, _ = model(imgs)
            # concept, _, _ = model(imgs)
            concept = torch.cat(concept, dim=1)
        concepts.append(concept)

    concepts = torch.cat(concepts, dim=0)
    return concepts

def evaluate_recall(model: FuseCBM, dataloader, device, values=None):
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

            if values is None:
                embeddings = model.get_fused_embedding(imgs, return_concepts=False, return_extra_dim=False)
            else:
                embeddings = model.get_fused_embedding_with_intervention(imgs, attrs, return_concepts=False, return_extra_dim=False, intervention_values=values)
            embeddings = F.normalize(embeddings, dim=1)

            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    recall_list, num_rec = recall(embeddings, labels, rank=[1,5,10], ret_num=True)
    return embeddings, labels, recall_list, num_rec

def collect_embeddings_with_probs(model: FuseCBM, dataloader, device, values):
    """Collect embeddings from a FuseCBM model on the data loader with different probabilities

    Args:
        model (FuseCBM): FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on
        values (torch.Tensor): Values to intervene on

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: List of embeddings, List of labels
    """
    model.eval()
    model.to(device)

    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 1.0]
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for prob in probs:
            prob_embeddings = []
            prob_labels = []
            for imgs, attrs, labels in tqdm(dataloader):
                imgs = imgs.to(device)
                attrs = attrs.to(device)
                labels = labels.to(device)

                if values is None:
                    embeddings = model.get_fused_embedding(imgs, return_concepts=False, return_extra_dim=False)
                else:
                    embeddings = model.get_fused_embedding_with_prob(imgs, attrs, return_concepts=False, intervention_values=values, prob_correct=prob)
                    # embeddings = model.get_fused_embedding_with_percentage_correction(imgs, attrs, return_concepts=False, return_extra_dim=False, intervention_values=values, percentage_correction=prob)
                embeddings = F.normalize(embeddings, dim=1)

                prob_embeddings.append(embeddings)
                prob_labels.append(labels)

            embeddings_list.append(torch.cat(prob_embeddings, dim=0))
            labels_list.append(torch.cat(prob_labels, dim=0))

    return embeddings_list, labels_list


def evaluate_recall_with_probs(model: FuseCBM, dataloader, device, values=None, prob=1.0):
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
        for imgs, attrs, labels in (dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            if values is None:
                embeddings = model.get_fused_embedding(imgs, return_concepts=False, return_extra_dim=False)
            else:
                # embeddings = model.get_fused_embedding_with_prob(imgs, attrs, return_concepts=False, intervention_values=values, prob_correct=prob)
                embeddings = model.get_fused_embedding_with_percentage_correction(imgs, attrs, return_concepts=False, return_extra_dim=False, intervention_values=values, percentage_correction=prob)
            embeddings = F.normalize(embeddings, dim=1)

            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    recall_list, num_rec = recall(embeddings, labels, rank=[1,5,10], ret_num=True)
    return embeddings, labels, recall_list, num_rec

def evaluate_recall_with_probs_gallery(model: FuseCBM, dataloader, device, values=None, prob=1.0, gallery_features=None, gallery_labels=None):
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
        for imgs, attrs, labels in (dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            if values is None:
                embeddings = model.get_fused_embedding(imgs, return_concepts=False, return_extra_dim=False)
            else:
                embeddings = model.get_fused_embedding_with_prob(imgs, attrs, return_concepts=False, intervention_values=values, prob_correct=prob)
            embeddings = F.normalize(embeddings, dim=1)

            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    recall_list, num_rec = recall(embeddings, labels, rank=[1,5,10], ret_num=True, gallery_features=gallery_features, gallery_labels=gallery_labels)
    return embeddings, labels, recall_list, num_rec

