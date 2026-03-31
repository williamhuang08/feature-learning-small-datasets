import argparse
import os
import math
from re import X
import sys
from pathlib import Path

import numpy as np
import tools
import wandb

import torch
import torch.nn as nn
from nn.models.nn import NN
from nn.utils.utils import save_nn_checkpoint
import torch.optim as optim



parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "result.log", type = str, help = "Output File")
parser.add_argument('-max_tot', default = 5000, type = int, help = "Maximum number of data samples")
parser.add_argument('-max_dep', default = 5, type = int, help = "Maximum number of depth")


args = parser.parse_args()

MAX_N_TOT = args.max_tot
datadir = args.dir

outf = open(args.file, "w")
print ("Dataset\tValidation Acc\tTest Acc", file = outf)

for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    """ PREPARING THE DATASET """
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])

    if c != 2:
        continue

    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    
    if n_tot > MAX_N_TOT or n_test > 0:
        print (str(dataset) + '\t0\t0', file = outf)
        continue

    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

    # load training and validation set
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]
    
    dataset_path = os.path.join(datadir, dataset)
    arff_file = os.path.join(dataset_path, f"{dataset}.arff") # data in the arff file

    rows = []
    in_data = False

    """ LOAD IN THE DATA """
    with open(arff_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data:
                rows.append(list(map(float, line.split(","))))

    all_data = torch.tensor(rows, dtype=torch.float32)

    X = all_data[:, :-1] # first 6 columns
    y = all_data[:, -1].long() # last column = class label

    X_train = X[train_fold]
    y_train = y[train_fold]
    X_val = X[val_fold]
    y_val = y[val_fold]

    """ TRAINING THE MODEL """

    global_best_lr = 0.0
    global_best_batch_norm = None
    global_best_num_layers = 0
    global_best_val_loss = float('inf')

    lrs = [0.1, 1]
    batch_norms = [True, False]
    num_layers_poss= [1, 2, 3, 4, 5]
    num_epochs = 500


    # grid search of parameters
    for lr in lrs:
        for batch_norm in batch_norms:
            for num_layers in num_layers_poss:
                wandb.init(
                    project="feature-learning-small-datasets",
                    name=f"dataset_{dataset}_lr_{lr}_batch_norm_{batch_norm}_num_layers_{num_layers}",
                    config={
                        "lr": lr,
                        "batch_norm": batch_norm,
                        "num_layers": num_layers,
                        "dataset": dataset,
                        "num_classes": c,
                        "num_features": d,
                        "num_epochs": num_epochs,
                    },
                )
                model = NN(num_layers, d, 512, c, batch_norm)
                loss_fn = nn.BCEWithLogitsLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr) # SGD optimizer
                
                best_val_loss = float('inf')
                train_losses, val_losses = [], []

                for epoch in range(1, num_epochs + 1):
                    optimizer.zero_grad()
                    out = model(X_train)
                    loss = loss_fn(out, y_train)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())

                    val_out = model(X_val)
                    val_loss = loss_fn(val_out, y_val)

                    val_losses.append(val_loss.item())

                    print(f"Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    log_dict = {
                        "train/loss": loss.item(),
                        "val/loss": val_loss.item(),
                        "epoch": epoch,
                    }
                    wandb.log(log_dict)

                if best_val_loss < global_best_val_loss:
                    global_best_val_loss = best_val_loss
                    global_best_lr = lr
                    global_best_batch_norm = batch_norm
                    global_best_num_layers = num_layers

                    save_nn_checkpoint(f"nn/results/dataset_{dataset}/model_weights/lr_{lr}_batch_norm_{batch_norm}_num_layers_{num_layers}.pth", model)
                wandb.finish()

    layer_1_weights = model.model[0].weight.detach().numpy()
    nfm = compute_nfm(layer_1_weights)
    agop = compute_agop(layer_1_weights)

    np.save('nn/results/dataset_{dataset}/matrices/nfm.npy', nfm)
    np.save('nn/results/dataset_{dataset}/matrices/agop.npy', agop)



