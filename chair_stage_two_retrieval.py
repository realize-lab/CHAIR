from obiwan.new_models import CBM, FuseCBM, MultiFuse, LambdaFuseCBM, Plain
from obiwan.datasets.cub import get_cub_dataloaders
from obiwan.datasets.awa import get_awa_dataloaders
from obiwan.utils import recall
import evaluate as ev
import intervene as iv
from torchmetrics.classification import Accuracy

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchmetrics.aggregation import MeanMetric

import os
import random
from dotenv import load_dotenv
import json
load_dotenv()

import wandb #noqa

try:
    from rich.tqdm import tqdm
except ImportError:
    from tqdm import tqdm


def get_concepts(model, dataloader, device) -> torch.Tensor:
    """Get the concept embeddings for the entire dataset

    Args:
        model (CBM): CBM or FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on

    Returns:
        torch.Tensor: Concept embeddings for the entire dataset
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


def collect_embeddings_with_probs(model: FuseCBM, dataloader, device, values):
    """Collect embeddings for the entire dataset with different probabilities of random intervention

    Args:
        model (FuseCBM): FuseCBM model
        dataloader (DataLoader): DataLoader object
        device (str): Device to run the model on
        values (torch.Tensor): Concept values for intervention (95th percentile)

    Returns:
        List[torch.Tensor]: List of embeddings for different probabilities

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
            for batch in tqdm(dataloader):
                if len(batch) == 2:
                    imgs, (labels, attrs) = batch
                else:
                    imgs, attrs, labels = batch

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

def train_sequential(model, train_loader, val_loader, num_concepts, device, epochs, lr, weight_decay, num_classes):
    """Train concept bottleneck and classification sequentially

    Args:
        model (FuseCBM): FuseCBM model
        train_loader (DataLoader): DataLoader object for training
        val_loader (DataLoader): DataLoader object for validation
        num_concepts (int): Number of concepts
        device (str): Device to run the model on
        epochs (int): Number of epochs
        lr (float): Learning rate
        weight_decay (float): Weight decay
        num_classes (int): Number of classes
    """

    optimizer = torch.optim.SGD(model.get_concept_parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Concept Bottleneck Training
    for epoch in tqdm(range(epochs)):
        model.set_concepts_to_train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for data in tqdm(train_loader):
            if len(data) == 2:
                imgs, (labels, attrs) = data
            else:
                imgs, attrs, labels = data
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            concepts_loss = 0
            criterion = torch.nn.CrossEntropyLoss()
            
            for i in range(num_concepts):
                ind_concept_loss = criterion(concepts[i].squeeze(), attrs[:,i].squeeze().float())
                concepts_loss = concepts_loss + ind_concept_loss

            concepts_loss = concepts_loss / num_concepts
            loss = concepts_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'concept_loss': loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.set_concepts_to_train()

    # Classification Training
    model.set_classes_to_train()
    optimizer = torch.optim.SGD(model.get_class_parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        model.set_classes_to_train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for data in tqdm(train_loader):
            if len(data) == 2:
                imgs, (labels, attrs) = data
            else:
                imgs, attrs, labels = data
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            class_loss = torch.nn.functional.cross_entropy(pred_classes, labels.long().squeeze())
            loss = class_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'class_loss': loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.set_classes_to_train()
            

def train_joint(model, train_loader, val_loader, num_concepts, device, epochs, lr, weight_decay, num_classes):
    """Train concept bottleneck and classification jointly

    Args:
        model (FuseCBM): FuseCBM model
        train_loader (DataLoader): DataLoader object for training
        val_loader (DataLoader): DataLoader object for validation
        num_concepts (int): Number of concepts
        device (str): Device to run the model on
        epochs (int): Number of epochs
        lr (float): Learning rate
        weight_decay (float): Weight decay
        num_classes (int): Number of classes
    """

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for data in tqdm(train_loader):
            if len(data) == 2:
                imgs, (labels, attrs) = data
            else:
                imgs, attrs, labels = data
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            class_loss = torch.nn.functional.cross_entropy(pred_classes, labels.long().squeeze())
            concepts_loss = 0
            criterion = torch.nn.CrossEntropyLoss()
            
            for i in range(num_concepts):
                ind_concept_loss = criterion(concepts[i].squeeze(), attrs[:,i].squeeze().float())
                concepts_loss = concepts_loss + ind_concept_loss

            concepts_loss = concepts_loss / num_concepts
            loss = class_loss + concepts_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'class_loss': loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.train()


@hydra.main(config_path="configs", config_name="vanilla", version_base="1.1")
def train(cfg: DictConfig) -> None:
    wandb.init(project=os.environ.get('WANDB_PROJECT', 'CHAIR'), entity=os.environ.get('WANDB_ENTITY'), config=OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    ))

    print(f"Dataset: {cfg.dataset} Seed: {cfg.seed} Mode: {cfg.train_mode}")

    seed = cfg.get('seed', random.randint(0, 10000))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.dataset == 'cub':
        train_loader, val_loader = get_cub_dataloaders("/nfs/turbo/coe-ecbk/vballoli/ConceptRetrieval/cem/cem/data/CUB200/class_attr_data_10/", cfg.batch_size, cfg.num_workers)
        num_classes = 100
        num_concepts = 112
    elif cfg.dataset == 'awa':
        train_loader, val_loader = get_awa_dataloaders(cfg.batch_size, cfg.num_workers)
        num_classes = 50
        num_concepts = 45
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else device

    backbone = resnet18(pretrained=cfg.pretrained)

    model = FuseCBM(backbone, num_classes, num_concepts, 0.2, 0, 'relu')

    model.to(device)

    exp_name = f"fuse_stage_two_retrieval_{cfg.dataset}_{cfg.seed}_{cfg.train_mode}"

    results_dir = "new_results"
    os.makedirs(results_dir, exist_ok=True)

    results_exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(results_exp_dir, exist_ok=True)

    # Concept Bottleneck Training
    if cfg.train_mode == 'sequential':
        train_sequential(model, train_loader, val_loader, num_concepts, device, cfg.epochs, cfg.lr, cfg.weight_decay, num_classes)
    elif cfg.train_mode == 'joint':
        train_joint(model, train_loader, val_loader, num_concepts, device, cfg.epochs, cfg.lr, cfg.weight_decay, num_classes)
    else:
        raise ValueError(f"Unknown training mode: {cfg.train_mode}")

    # CHAIR training - re-initialize classification layer and train the projection layer
    model.reset_classification_2()
    class_parameters = model.get_class_parameters()
    optimizer = torch.optim.SGD(class_parameters, lr=1e-3, momentum=0.9, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    model.to(device)
    concepts = iv.get_concepts(model, train_loader, device).cpu().numpy()
    concepts_max = np.percentile(concepts, 95, axis=0).astype(np.float32)
    concepts_max = torch.from_numpy(concepts_max).to(device)

    for epoch in tqdm(range(cfg.epochs)):
            epoch_loss_classes = MeanMetric()
            epoch_loss_classes.to(device)

            class_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
            class_accuracy.to(device)

            for data in tqdm(train_loader):
                if len(data) == 2:
                    imgs, (labels, attrs) = data
                else:
                    imgs, attrs, labels = data

                imgs = imgs.to(device)
                attrs = attrs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                prob = torch.rand(1).item()
                # Fused embedding - merging previous embedding and edits from the projection layer
                fused = model.get_fused_embedding_with_prob(imgs, attrs, False, False, concepts_max, prob)
                classes = model.get_class_from_embedding(fused)

                
                loss = torch.nn.functional.cross_entropy(classes, labels.long().squeeze())
                loss.backward()
                optimizer.step()

                epoch_loss_classes.update(loss)
                wandb.log({'class_loss': loss})

            print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")
            wandb.log({"epoch_class_loss": epoch_loss_classes.compute()})
            lr_scheduler.step()

            for imgs, attrs, labels in tqdm(train_loader):
                imgs = imgs.to(device)
                attrs = attrs.to(device)
                labels = labels.to(device)

                concepts, classes = model(imgs)
                class_accuracy.update(classes, labels.long().squeeze())

            train_class_accuracy = class_accuracy.compute()
            wandb.log({'new_train_class_accuracy': train_class_accuracy})

            recall_list = ev.evaluate_recall(model, val_loader, device)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})
    
    torch.save(model.state_dict(), os.path.join(results_exp_dir, 'model.pth'))

    recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)

    # with open(os.path.join(results_exp_dir, 'results.json'), 'w') as f:
    #     json.dump({'recall@1': recall_list[0], 'recall@5': recall_list[1], 'recall@10': recall_list[2]}, f)
    results = {'recall@1': recall_list[0], 'recall@5': recall_list[1], 'recall@10': recall_list[2]}

    print(f'Recall@1: {recall_list[0]} - Recall@5: {recall_list[1]} - Recall@10: {recall_list[2]}')

    concepts = get_concepts(model, train_loader, device).cpu().numpy()

    # get 95% and 5% quantiles for each concept for intervention
    # concept_min = torch.quantile(concepts, 0.05, dim=0)
    # concept_max = torch.quantile(concepts, 0.95, dim=0)
    concepts_min = np.percentile(concepts, 5, axis=0).astype(np.float32)
    concepts_max = np.percentile(concepts, 95, axis=0).astype(np.float32)
    

    prob_embs, prob_labels = collect_embeddings_with_probs(model, val_loader, device, torch.from_numpy(concepts_max).to(device).squeeze())

    results['prob_results'] = {}

    for i in range(len(prob_embs)):
        for j in range(len(prob_embs)):
            print(f"Prob: {i} vs Prob: {j}")
            recall_list, num_rec = recall(prob_embs[i], prob_labels[i], rank=[1,5,10], gallery_features=prob_embs[j], gallery_labels=prob_labels[j], ret_num=True)
            print(f"{recall_list}\t{num_rec}")

            results['prob_results'][i,j] = {'recall': recall_list, 'num_rec': num_rec}

    with open(os.path.join(results_exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f)



if __name__ == '__main__':
    train()
