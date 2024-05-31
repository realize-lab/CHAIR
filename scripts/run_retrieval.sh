#!/bin/sh

# Replace script.py with name of your script and config.yaml with name of your config file

python script.py --config-name config.yaml +seed=$1 +dataset=$2 ++train_mode=$3