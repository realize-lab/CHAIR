import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet


def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'softmax':
        return nn.Softmax(dim=1)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation function: {name}')

class Plain(nn.Module):
    """Simple ResNet model for classification (no concepts)"""

    def __init__(self, backbone: ResNet, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout)

        self.class_prediction_layers = nn.Sequential(*[
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        classes = self.class_prediction_layers(x)

        return classes
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        x = self.dropout(x)
        
        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)
        
        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        embedding = x
        
        return embedding
    
    def get_fused_embedding(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.get_embedding(x)

class CBM(nn.Module):
    """ResNet model with CBM like architecture"""

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.dropout_prob = dropout
        self.extra_dim = extra_dim

        self.concept_layer = nn.ModuleList([nn.Linear(self.backbone.fc.in_features, 1) for _ in range(num_concepts)])
        self.concept_activation = get_activation(concept_activation)
        self.class_layer = nn.Linear(num_concepts, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        if self.extra_dim > 0:
            self.extra_dim_layer = nn.Linear(self.backbone.fc.in_features, extra_dim)

    def forward(self, x: torch.Tensor, ret_emb: bool=False):
        x = self.get_embedding(x)
        emb = x
        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts_for_class = torch.cat(concepts, dim=1)
        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            concepts_for_class = torch.cat([concepts_for_class, extra_dim], dim=1)

        x = self.class_layer(concepts_for_class)

        if ret_emb:
            return concepts, x, emb

        return concepts, x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_concepts(self, x: torch.Tensor, get_extra: bool=False):
        x = self.get_embedding(x)
        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        if get_extra and self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            return concepts, extra_dim
        return concepts
    
    def get_class(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_dim > 0:
            concepts, extra_dim = self.get_concepts(x, get_extra=True)
            x = self.class_layer(torch.cat([torch.cat(concepts, dim=1), extra_dim], dim=1))
        else:
            concepts = self.get_concepts(x)
            x = self.class_layer(concepts)
        return x
    
    def get_class_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        x = self.class_layer(concepts)
        return x

    def get_concept_parameters(self):
        return list(self.backbone.parameters()) + list(self.concept_layer.parameters())
    
    def get_class_parameters(self):
        return list(self.class_layer.parameters())

    def set_concepts_to_train(self):
        for param in self.concept_layer.parameters():
            param.requires_grad = True
        for param in self.backbone.parameters():
            param.requires_grad = True

        for param in self.class_layer.parameters():
            param.requires_grad = False

    def set_classes_to_train(self):
        for param in self.concept_layer.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.class_layer.parameters():
            param.requires_grad = True

    def get_class_after_intervention(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_intervention: float=1.0) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts = torch.cat(concepts, dim=1)
        # intervened concepts are attr converted to one hot
        # intervened_concepts = torch.zeros_like(concepts)
        
        if intervention_values is not None:
            if prob_intervention != 1.0:
                concepts_to_intervene_on_for_batch = (torch.rand_like(concepts) < prob_intervention).bool() 
                concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch] * attr[concepts_to_intervene_on_for_batch]
            else:
                concepts = intervention_values * attr
        else:
            concepts[attr.long()] = 0.95
            concepts[~attr.long()] = 0.05

        return self.get_class_from_concepts(concepts)

    def get_fused_embedding(self, x: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, ret_emb:bool=False) -> torch.Tensor:
        return self.get_embedding(x)
    


class HybridCBM(CBM):
    """ResNet model with CBM like architecture and extra dimension - Mahinpei et al. 2021"""

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str):
        super().__init__(backbone, num_classes, num_concepts, dropout, extra_dim, concept_activation)

        self.class_layer = nn.Linear(num_concepts + extra_dim, num_classes)

    def forward(self, x: torch.Tensor, ret_emb: bool=False):
        x = self.get_embedding(x)
        emb = x
        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts_for_class = torch.cat(concepts, dim=1)
        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            concepts_for_class = torch.cat([concepts_for_class, extra_dim], dim=1)

        x = self.class_layer(concepts_for_class)

        if ret_emb:
            return (concepts, extra_dim), x, emb

        return (concepts, extra_dim), x
    
    def get_class(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_dim > 0:
            concepts, extra_dim = self.get_concepts(x, get_extra=True)
            x = self.class_layer(torch.cat([torch.cat(concepts, dim=1), extra_dim], dim=1))
        else:
            concepts = self.get_concepts(x)
            x = self.class_layer(concepts)
        return x
    
    def get_class_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        x = self.class_layer(concepts)
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_hybrid_embedding(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_intervention: float=1.0) -> torch.Tensor:
        x = self.get_embedding(x)
        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts = torch.cat(concepts, dim=1)
        
        if intervention_values is not None:
            concepts_to_intervene_on_for_batch = (torch.rand_like(concepts) < prob_intervention).bool() 
            concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch] * attr[concepts_to_intervene_on_for_batch]

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            concepts = torch.cat([concepts, extra_dim], dim=1)
        return concepts
    
    def set_concepts_to_train(self):
        for param in self.concept_layer.parameters():
            param.requires_grad = True
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.extra_dim_layer.parameters():
            param.requires_grad = True

        for param in self.class_layer.parameters():
            param.requires_grad = False

    def set_classes_to_train(self):
        for param in self.concept_layer.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.extra_dim_layer.parameters():
            param.requires_grad = False

        for param in self.class_layer.parameters():
            param.requires_grad = True


class FuseCBM(CBM):
    """Our CHAIR model with CBM like architecture and Fusion Head"""

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str):
        super().__init__(backbone, num_classes, num_concepts, dropout, extra_dim, concept_activation)

        self.class_layer = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.fuse_layer = nn.Linear(num_concepts+extra_dim, self.backbone.fc.in_features)
        
    def get_fused_embedding(self, x: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, ret_emb:bool=False) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # embedding
        emb = x

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        # fuse layer takes in the concepts and returns an "edit" to the embedding 
        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
        
            fused = self.fuse_layer(torch.cat([torch.cat(concepts, dim=1), extra_dim], dim=1))
        else:
            fused = self.fuse_layer(torch.cat(concepts, dim=1))
            # fused = torch.nn.ReLU()(fused)

        # simple addition of the fused embedding to the original embedding
        fused = x + fused

        if return_concepts:
            if ret_emb:
                return fused, concepts, emb
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def forward(self, x: torch.Tensor, ret_emb: bool=False):
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding(x, return_concepts=True, return_extra_dim=True)
        else:
            fused, concepts, emb = self.get_fused_embedding(x, return_concepts=True, ret_emb=True)

        x = self.class_layer(fused)
        if ret_emb:
            return concepts, x, emb
        return concepts, x
    
    def set_concepts_to_train(self):
        super().set_concepts_to_train()
        for param in self.fuse_layer.parameters():
            param.requires_grad = False

    def set_classes_to_train(self):
        super().set_classes_to_train()
        for param in self.fuse_layer.parameters():
            param.requires_grad = True

    def set_fuse_to_train(self):
        for param in self.concept_layer.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.class_layer.parameters():
            param.requires_grad = False

        for param in self.fuse_layer.parameters():
            param.requires_grad = True

    def get_concept_parameters(self):
        return super().get_concept_parameters()
    
    def get_class_parameters(self):
        return super().get_class_parameters() + list(self.fuse_layer.parameters())
    
    def get_class(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding(x, return_concepts=True, return_extra_dim=True)
        else:
            fused, concepts = self.get_fused_embedding(x, return_concepts=True)
        x = self.class_layer(fused)
        return x
    
    def get_fused_embedding_with_intervention(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None) -> torch.Tensor:
        """Get the fused embedding with intervention on the concepts

        Args:
            x (torch.Tensor): input image
            attr (torch.Tensor): attributes to intervene on
            return_concepts (bool): whether to return the concepts
            return_extra_dim (bool, optional): whether to return the extra dimension. Defaults to False.
            intervention_values (torch.Tensor, optional): values to intervene on. Defaults to None.

        Returns:
            torch.Tensor: fused embedding
        """
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts = torch.cat(concepts, dim=1)
        # intervened concepts are attr converted to one hot
        # intervened_concepts = torch.zeros_like(concepts)
        
        if intervention_values is not None:
            concepts = intervention_values * attr
        else:
            concepts[attr.long()] = 0.95
            concepts[~attr.long()] = 0.05

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            intervened_extra_dim = attr[:, -1]

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

        fused = x + fused

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def get_class_after_intervention(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_intervention: float=1.0) -> torch.Tensor:
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding_with_intervention(x, attr, return_concepts=True, return_extra_dim=True, intervention_values=intervention_values)
        else:
            fused, concepts = self.get_fused_embedding_with_prob(x, attr, return_concepts=True, intervention_values=intervention_values, prob_correct=prob_intervention)
        x = self.class_layer(fused)
        return x

    def get_fused_embedding_with_prob(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_correct: float=0.5, neg: bool=False) -> torch.Tensor:
        """Get the fused embedding with probability of intervention on the concepts"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)

        if neg:
            attr = 1 - attr

        intervention_values = intervention_values.repeat(attr.shape[0], 1)
        # select random concepts with probability prob_correct to intervene on
        if prob_correct <= 1.:
            concepts_to_intervene_on_for_batch = torch.arange(concepts.shape[1])[:int(concepts.shape[1]*prob_correct)]
            indices = torch.zeros_like(concepts)
            indices[:, concepts_to_intervene_on_for_batch] = 1
            concepts_to_intervene_on_for_batch = indices.bool()
            torch.where(attr[concepts_to_intervene_on_for_batch] == 1, intervention_values[concepts_to_intervene_on_for_batch], 0)
            concepts[concepts_to_intervene_on_for_batch] = attr[concepts_to_intervene_on_for_batch].float()
        else:
            concepts = attr * intervention_values

        # set these intervened concepts to their corresponding intervention values
        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

            fused = fused + x

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def get_fused_embedding_with_group_prob_correction(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, percentage_groups: float=0.5, group2concept: torch.Tensor=None) -> torch.Tensor:
        """Get the fused embedding with group probability correction"""
        assert group2concept is not None, "group2concept is required for group correction"

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)


        intervention_values = intervention_values.repeat(attr.shape[0], 1)
        # select the indices from the group2concept tensor
        if percentage_groups <= 1.:
            groups_to_intervene_on = torch.randperm(group2concept.shape[0])[:int(group2concept.shape[0]*percentage_groups)]
            concepts_to_intervene_on_for_batch = torch.zeros(group2concept.shape[1], dtype=group2concept.dtype)
            for group in groups_to_intervene_on:
                concepts_to_intervene_on_for_batch += group2concept[group]

            concepts_to_intervene_on_for_batch = concepts_to_intervene_on_for_batch.bool()
            indices = torch.zeros_like(concepts)
            indices[:, concepts_to_intervene_on_for_batch] = 1
            concepts_to_intervene_on_for_batch = indices.bool()
            torch.where(attr[concepts_to_intervene_on_for_batch] == 1, intervention_values[concepts_to_intervene_on_for_batch], 0)
            concepts[concepts_to_intervene_on_for_batch] = attr[concepts_to_intervene_on_for_batch].float()
            
        else:
            concepts = attr * intervention_values


        # set these intervened concepts to their corresponding intervention values
        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

            fused = fused + x

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def get_fused_embedding_with_group_correction(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, group2concept: torch.Tensor=None, group_id: int=0) -> torch.Tensor:
        """Get the fused embedding with group intervention on the concepts"""
        assert group2concept is not None, "group2concept is required for group correction"

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)

        # select the indices from the group2concept tensor
        concepts_to_intervene_on_for_batch = group2concept[group_id]
        indices = torch.zeros_like(concepts)
        indices[:, concepts_to_intervene_on_for_batch] = 1
        concepts_to_intervene_on_for_batch = indices.bool()
        torch.where(attr[concepts_to_intervene_on_for_batch] == 1, intervention_values[concepts_to_intervene_on_for_batch], 0)
        concepts[concepts_to_intervene_on_for_batch] = attr[concepts_to_intervene_on_for_batch].float()
        # concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch] * attr[concepts_to_intervene_on_for_batch]

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

            fused = fused + x

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused

    def get_fused_embedding_with_percentage_correction(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, percentage_correction: float=0.5) -> torch.Tensor:
        """Get the fused embedding with percentage of concepts intervened on"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)

        intervention_values = intervention_values.repeat(attr.shape[0], 1)
        # get concepts whose delta from the ground truth is greater than 0.5
        concepts_to_intervene_on_for_batch = torch.abs(concepts - attr) > 0.5
        # set a percentage of these intervened concepts to their corresponding intervention values
        concepts_to_intervene_on_for_batch = concepts_to_intervene_on_for_batch & (torch.rand_like(concepts) < percentage_correction).bool()
        concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch]

        fused = self.fuse_layer(concepts)
        fused = fused + x

        if return_concepts:
            return fused, concepts
        
        return fused
        
    
    def get_class_from_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.class_layer(x)
        return x
    
    def reset_classification(self) -> torch.Tensor:
        self.class_layer = nn.Linear(self.backbone.fc.in_features, self.num_classes)

    def reset_classification_2(self) -> torch.Tensor:
        self.class_layer = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        self.fuse_layer = nn.Linear(self.num_concepts, self.backbone.fc.in_features)

    def add_extra(self):
        self.extra_class = nn.Linear(self.num_concepts, self.num_classes)

    def extra_forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, concepts, emb = self.get_fused_embedding(x, return_concepts=True, ret_emb=True)
        x = self.extra_class(torch.cat(concepts, dim=1))
        return concepts, x
    
    def get_extra_class_after_intervention(self, img: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_intervention: float=1.0) -> torch.Tensor:
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding_with_intervention(img, attr, return_concepts=True, return_extra_dim=True, intervention_values=intervention_values)
        else:
            fused, concepts = self.get_fused_embedding_with_prob(img, attr, return_concepts=True, intervention_values=intervention_values, prob_correct=prob_intervention)
        x = self.extra_class(torch.cat(concepts, dim=1))
        return x


class MultiFuse(FuseCBM):
    """CHAIR model with two classification heads - one before extracting the concepts and one at the end"""

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str):
        super().__init__(backbone, num_classes, num_concepts, dropout, extra_dim, concept_activation)

        self.extra_class_layer = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor):
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding(x, return_concepts=True, return_extra_dim=True)
        else:
            fused, concepts = self.get_fused_embedding(x, return_concepts=True)

        x = self.class_layer(fused)
        extra_x = self.extra_class_layer(torch.cat(concepts, dim=1))
        return concepts, x, extra_x
    
    def get_class_from_concepts(self, concepts: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        x = self.class_layer(fused)
        extra_x = self.extra_class_layer(concepts)
        return x, extra_x
    
    def get_class_after_intervention(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool = False, intervention_values: torch.Tensor = None, prob_intervention: float=1.0) -> torch.Tensor:
        if self.extra_dim > 0:
            fused, concepts, extra_dim = self.get_fused_embedding_with_intervention(x, attr, return_concepts=True, return_extra_dim=True, intervention_values=intervention_values)
        else:
            fused, concepts = self.get_fused_embedding_with_prob(x, attr, return_concepts=True, intervention_values=intervention_values, prob_correct=prob_intervention)
        x, extra_x = self.get_class_from_concepts(concepts, fused)
        return x
    

class LambdaFuseCBM(FuseCBM):
    """CHAIR model with variable fusion weights for merging the embeddings and the edits"""

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str, lambda_: float):
        super().__init__(backbone, num_classes, num_concepts, dropout, extra_dim, concept_activation)

        self.lambda_ = lambda_

    def get_fused_embedding(self, x: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, ret_emb:bool=False) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # embedding
        emb = x

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
        
            fused = self.fuse_layer(torch.cat([torch.cat(concepts, dim=1), extra_dim], dim=1))
        else:
            fused = self.fuse_layer(torch.cat(concepts, dim=1))
            # fused = torch.nn.ReLU()(fused)

        fused = (1-self.lambda_) * x + self.lambda_ * fused

        if return_concepts:
            if ret_emb:
                return fused, concepts, emb
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def get_fused_embedding_with_intervention(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts = torch.cat(concepts, dim=1)
        # intervened concepts are attr converted to one hot
        # intervened_concepts = torch.zeros_like(concepts)
        
        if intervention_values is not None:
            concepts = intervention_values * attr
        else:
            concepts[attr.long()] = 0.95
            concepts[~attr.long()] = 0.05

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            intervened_extra_dim = attr[:, -1]

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

        fused = (1-self.lambda_) * x + self.lambda_ * fused

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused
    
    def get_fused_embedding_with_prob(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, prob_correct: float=0.5, neg: bool=False) -> torch.Tensor:
        """Get the fused embedding with probability of intervention on the concepts"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)

        if neg:
            attr = 1 - attr

        intervention_values = intervention_values.repeat(attr.shape[0], 1)
        # select random concepts with probability prob_correct to intervene on
        if prob_correct != 1.0:
            # concepts_to_intervene_on_for_batch = (torch.rand_like(concepts) < prob_correct).bool() 
            concepts_to_intervene_on_for_batch = torch.randperm(concepts.shape[1])[:int(concepts.shape[1]*prob_correct)]   
            indices = torch.zeros_like(concepts)
            indices[:, concepts_to_intervene_on_for_batch] = 1
            concepts_to_intervene_on_for_batch = indices.bool()
            concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch] * attr[concepts_to_intervene_on_for_batch]
        else:
            concepts = attr * intervention_values
        # set these intervened concepts to their corresponding intervention values

        if self.extra_dim > 0:
            extra_dim = self.extra_dim_layer(x)
            fused = self.fuse_layer(concepts)

            fused = fused + self.fuse_layer(concepts)

        else:
            fused = self.fuse_layer(concepts)

            fused = (1-self.lambda_) * x + self.lambda_ * fused

        if return_concepts:
            if return_extra_dim:
                return fused, concepts, extra_dim
            return fused, concepts
        
        return fused

    def get_fused_embedding_with_percentage_correction(self, x: torch.Tensor, attr: torch.Tensor, return_concepts: bool, return_extra_dim: bool=False, intervention_values: torch.Tensor=None, percentage_correction: float=0.5) -> torch.Tensor:
        """Get the fused embedding with percentage of concepts intervened on"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]

        concepts = torch.cat(concepts, dim=1)

        intervention_values = intervention_values.repeat(attr.shape[0], 1)
        # get concepts whose delta from the ground truth is greater than 0.5
        concepts_to_intervene_on_for_batch = torch.abs(concepts - attr) > 0.5
        # set a percentage of these intervened concepts to their corresponding intervention values
        concepts_to_intervene_on_for_batch = concepts_to_intervene_on_for_batch & (torch.rand_like(concepts) < percentage_correction).bool()
        concepts[concepts_to_intervene_on_for_batch] = intervention_values[concepts_to_intervene_on_for_batch]

        fused = self.fuse_layer(concepts)
        fused = (1-self.lambda_) * x + self.lambda_ * fused

        if return_concepts:
            return fused, concepts
        
        return fused