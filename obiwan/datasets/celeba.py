# Credit: https://github.com/mateoespinosa/cem
import logging
import numpy as np
import os
import torch
import torchvision

from pathlib import Path
from pytorch_lightning import seed_everything
from torchvision import transforms


###############################################################################
## GLOBAL VARIABLES
###############################################################################


# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
assert os.environ.get("DATASET_DIR") is not None, "Please set the environment variable DATASET_DIR to the path of the CelebA dataset"
DATASET_DIR = os.environ.get("DATASET_DIR")



#########################################################
## CONCEPT INFORMATION REGARDING CelebA
#########################################################


SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################


def generate_data(
    use_binary_vector_class=True,
    label_binary_width=5,
    num_concepts=40,
    num_classes=1000,
    num_hidden_concepts=0,
    label_dataset_subsample=1,
    selected_concepts=False,
    result_dir=None,
    image_size=64,
    batch_size=512,
    num_workers=8,
    weight_loss=False,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    concept_group_map = None
    # seed_everything(seed)
    # use_binary_vector_class = config.get('use_binary_vector_class', False)
    if use_binary_vector_class:
        # Now reload by transform the labels accordingly
        width = label_binary_width
        # width = config.get('label_binary_width', 5)
        def _binarize(concepts, selected, width):
            result = []
            binary_repr = []
            concepts = concepts[selected]
            for i in range(0, concepts.shape[-1], width):
                binary_repr.append(
                    str(int(np.sum(concepts[i : i + width]) > 0))
                )
            return int("".join(binary_repr), 2)
        
        def shift(x):
            return x[0].long() - 1

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            target_transform=shift,
            target_type=['attr'],
        )

        def first(x):
            return x[0]
        
        def second(x):
            return x[1]

        concept_freq = np.sum(
            celeba_train_data.attr.cpu().detach().numpy(),
            axis=0
        ) / celeba_train_data.attr.shape[0]
        logging.debug(f"Concept frequency is: {concept_freq}")
        sorted_concepts = list(map(
            first,
            sorted(enumerate(np.abs(concept_freq - 0.5)), key=second),
        ))
        # num_concepts = config.get(
        #     'num_concepts',
        #     celeba_train_data.attr.shape[-1],
        # )
        num_concepts = celeba_train_data.attr.shape[-1]
        concept_idxs = sorted_concepts[:num_concepts]
        concept_idxs = sorted(concept_idxs)
        if num_hidden_concepts > 0:
            num_hidden = num_hidden_concepts
            hidden_concepts = sorted(
                sorted_concepts[
                    num_concepts:min(
                        (num_concepts + num_hidden),
                        len(sorted_concepts)
                    )
                ]
            )
        else:
            hidden_concepts = []
        # logging.debug(f"Selecting concepts: {concept_idxs}")
        # logging.debug(f"\tAnd hidden concepts: {hidden_concepts}")
            
        def tt(x):
            return [
                torch.tensor(
                    _binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            ]
        
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=tt,
            target_type=['identity', 'attr'],
        )
        label_remap = {}
        vals, counts = np.unique(
            list(map(
                lambda x: _binarize(
                    x.cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts),
                    width=width,
                ),
                celeba_train_data.attr
            )),
            return_counts=True,
        )
        for i, label in enumerate(vals):
            label_remap[label] = i

        def tt_remap(x):
            return (
                torch.tensor(
                    label_remap[_binarize(
                        x[1].cpu().detach().numpy(),
                        selected=(concept_idxs + hidden_concepts),
                        width=width,
                    )],
                    dtype=torch.long,
                ),
                x[1][concept_idxs].float(),
            )

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=tt_remap,
            target_type=['identity', 'attr'],
        )
        num_classes = len(label_remap)

        # And subsample to reduce its massive size
        # factor = config.get('label_dataset_subsample', 1)
        factor = label_dataset_subsample
        if factor != 1:
            train_idxs = np.random.choice(
                np.arange(0, len(celeba_train_data)),
                replace=False,
                size=len(celeba_train_data)//factor,
            )
            logging.debug(f"Subsampling to {len(train_idxs)} elements.")
            celeba_train_data = torch.utils.data.Subset(
                celeba_train_data,
                train_idxs,
            )
    else:
        concept_selection = list(range(0, len(CONCEPT_SEMANTICS)))
        if selected_concepts:
            concept_selection = SELECTED_CONCEPTS

        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            target_transform=lambda x: x[0].long() - 1,
            target_type=['identity'],
        )
        num_concepts = celeba_train_data.attr.shape[-1]
        vals, counts = np.unique(
            celeba_train_data.identity,
            return_counts=True,
        )
        sorted_labels = list(map(
            lambda x: x[0],
            sorted(zip(vals, counts), key=lambda x: -x[1])
        ))
        # logging.debug(
        #     f"Selecting {config['num_classes']} out of {len(vals)} classes"
        # )
        # result_dir = config.get('result_dir', None)
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                os.path.join(
                    result_dir,
                    f"selected_top_{num_classes}_labels.npy",
                ),
                sorted_labels[:num_classes],
            )
        label_remap = {}
        for i, label in enumerate(sorted_labels[:num_classes]):
            label_remap[label] = i

        # Now reload by transform the labels accordingly
        celeba_train_data = torchvision.datasets.CelebA(
            root=root_dir,
            split='all',
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=lambda x: [
                torch.tensor(
                    # If it is not in our map, then we make it be the token
                    # label config['num_classes'] which will be removed
                    # afterwards
                    label_remap.get(
                        x[0].cpu().detach().item() - 1,
                        num_classes
                    ),
                    dtype=torch.long,
                ),
                x[1][concept_selection].float(),
            ],
            target_type=['identity', 'attr'],
        )
        num_classes = num_classes

        train_idxs = np.where(
            list(map(
                lambda x: x.cpu().detach().item() - 1 in label_remap,
                celeba_train_data.identity
            ))
        )[0]
        celeba_train_data = torch.utils.data.Subset(
            celeba_train_data,
            train_idxs,
        )
    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    logging.debug(
        f"Data split is: {total_samples} = {train_samples} (train) + "
        f"{test_samples} (test) + {val_samples} (validation)"
    )
    celeba_train_data, celeba_test_data, celeba_val_data = \
        torch.utils.data.random_split(
            celeba_train_data,
            [train_samples, test_samples, val_samples],
        )
    
    train_dl = torch.utils.data.DataLoader(
        celeba_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        celeba_test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        celeba_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Finally, determine whether or not we will need to compute the imbalance
    # factors
    if weight_loss:
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for i, (_, (y, c)) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (num_concepts, len(label_remap), concept_group_map),
    )
