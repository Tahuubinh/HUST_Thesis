# TA HUU BINH's HUST THESIS

This repository contains the source code of my bachelor's thesis at Hanoi University of Science and Technology.

## Introduction

The target is to find an effective offloading strategy in a vehicular edge-cloud computing system. I propose a new reinforcement learning method that handles specific attributes of the environment so that the system can minimize the total time surpassing the deadlines of tasks. Also, a technique of transfer learning can brings quality results given fewer training data than conventional approaches.

## Installation

For Windows, you can import the environment using conda:
```
conda env create -f PCWindows.yaml
```

For Ubuntu, you can import the environment using conda:
```
conda env create -f torch_ubuntu.yaml
```

You can also use `requirement.txt` to import the environment.

> Note: except for `PCWindows.yaml`, you need to install *argparse* module.

```
pip install argparse
```

## Data preparation

You can generate new data using [/code/config/random_task](https://github.com/Tahuubinh/HUST_Thesis/blob/main/code/config/random_task.py) or get the same data (which is in folder *data_task*) as in my thesis at:

[https://drive.google.com/file/d/1NKIxhExoFS9gBzuUkpZq64BTbpngv7-n/view?usp=sharing](https://drive.google.com/file/d/1NKIxhExoFS9gBzuUkpZq64BTbpngv7-n/view?usp=sharing)

Folder *data* contains preprocessed real [GPS data](https://kaggle.com/igorbalteiro/gps-data-from-rio-de-janeiro-buses) of buses in Rio de Janeiro, a state of Brazil in South America.

> Note: Both folders in the zip file should be put in the root of this repository! E.g. [./data_task/](https://github.com/Tahuubinh/HUST_Thesis/blob/main/data_task/)

## Reproducing experiments

To provide a flexible and powerful way to parse command-line arguments, I use *argparse* module. Open [/code/config/args_parser.py](https://github.com/Tahuubinh/HUST_Thesis/blob/main/code/config/args_parser.py) to see all arguments, their purposes, and default values. To run the experiments, you must be in [./code/](https://github.com/Tahuubinh/HUST_Thesis/blob/main/code/). All results will be saved in [./result/](https://github.com/Tahuubinh/HUST_Thesis/blob/main/result/).

### Quick run

Run AODAI in 30 independent times:

```
python main.py --algorithm='AODAI' --folder_name='AODAI' --num_times_to_run=30 --num_vec=5 --gamma=0.995 --target_model_update=0.01
```

Run Neural Network Crafting to transfer knowledge from a 5-VSs-to-10-VSs scenario:

```
python main.py --algorithm='transfer' --folder_name='TransferAODAI' --num_times_to_run=30 --num_vec=10 --typeTransfer='mid' --pre_numVSs=5
```
