# CHAIR - CBM-Enabled Human-AI Collaboration for Image Retrieval [IJCAI-24, CV4Animals@CVPR-24]

Code for the paper "Are They the Same Picture? Adapting Concept Bottleneck Models for Human-AI Collaboration in Image Retrieval" accepted at IJCAI-24 Human-Centered AI track and CV4Animals@CVPR-24 workshop.

Link to Paper webpage: ![CHAIR](https://realize-lab.github.io/CHAIR/)

Link to Paper PDF: Coming soon

## Setup

1. Create `.env` file in the root directory with the following content:
```
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=your_wandb_project_name
WANDB_ENTITY=your_wandb_entity
```
2. Install the required packages:
```
pip install -r requirements.txt
```

## Data

Follow the instructions for `CUB` from [ConceptBottleneck](https://github.com/yewsiang/ConceptBottleneck) and add the root directory to `data_dir` in the config file.

For `CelebA`, follow the instructions [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Note, the PyTorch vision download will most likely not work, so you will have to download the dataset manually and add the root directory to `data_dir` in the config file.

For `AwA2`, follow the instructions [https://cvml.ist.ac.at/AwA2/](https://cvml.ist.ac.at/AwA2/). Note, the PyTorch vision download will most likely not work, so you will have to download the dataset manually.

## OS Env

Set the following environment variables:
```
export AWA_DATA_DIR=/path/to/AwA2
export CUB_DATA_DIR=/path/to/CUB
export DATASET_DIR=/path/to/CelebA
```

## Scripts for SLURM

```bash
bash scripts/run_train.sh scripts/run_retrieval.sh JOB_NAME SEED DATASET_NAME TRAIN_MODE
```

Training modes: Sequential or Joint
Scripts: `chair_retrieval.py` or `chair_stage_two_retrieval.py` add to `scripts/run_retrieval.sh`
