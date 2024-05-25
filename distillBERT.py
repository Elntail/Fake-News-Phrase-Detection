#!/usr/bin/env python3

# CMPU 366 Final Project


import csv
import os
import sys
from typing import Callable, Tuple

import random as rd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

torch.manual_seed(0)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


EPOCHS = 20
BATCH_SIZE = 24
MAX_LENGTH = 10
LR = 0.001
CKPT_DIR = "./ckpt"
NUM_CLASSES = 2

N_DIMS = 768

label_map = {'0': 0, '1': 1, 'label': None}


class NN(nn.Module):
    def __init__(self, n_features: int):
        """Construct the pieces of the neural network."""

        # Initialize the parent class (nn.Module)
        super(NN, self).__init__()
        
        self.bert = bert

        self.hidden_layers = nn.Sequential(      

                            # Layer 1
                            nn.Linear(N_DIMS, N_DIMS // 2),
                            nn.ReLU(),

                            # Layer 2
                            nn.Linear(N_DIMS // 2, N_DIMS // 4),
                            nn.ReLU(),

                            # Layer 3
                            nn.Linear(N_DIMS // 4, N_DIMS // 8),
                            nn.ReLU(),

                            # Layer 4
                            nn.Linear(N_DIMS // 8, N_DIMS // 16),
                            nn.ReLU(),
                        
                            
        )
        self.flatten = nn.Flatten()
        # Reduce to the 2 labels (real or fake)
        # Layer 10

        self.output_layer = nn.Linear(N_DIMS * n_features // 16, 2)

        # Log probabilities of each class (specifying the dimension of the
        # input tensor to use)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """The forward pass of the model: Transform the input data x into
        the output predictions (the log probabilities for each label).
        """
        
        outputs = self.bert(x)
        last_hidden_states = outputs.last_hidden_state
        after_hidden = self.hidden_layers(last_hidden_states)
        flattened_states = self.flatten(after_hidden)
        scores = self.output_layer(flattened_states)
        probs = self.out(scores)
        
        return probs


####



def make_data(fname: str) -> Tuple[list[str], list[str], list[int]]:

    with open(fname, 'r', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='|')

        saved = ([], [], [])

        i = 0

        for row in data:

            # Gets title
            saved[0].append(row[1])

            # Gets text
            saved[1].append(row[2])

            # Gets label (real or fake)
            saved[2].append(label_map[row[3]])

            i += 1

            
            #threshold to make sure we don't use too much
            if i == 201 and fname == "train.csv":
                break

            if i == 201 and fname == "test.csv":
                break
            

        for list in saved:
            list.remove(list[0])

        return saved



def prep_bert_data(titles: list[str], max_length: int) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    padded_titles = [tokenizer(t, truncation= True, padding='max_length', max_length= max_length) for t in titles]

    #padded_texts = [tokenizer(t, truncation= True, padding='max_length', max_length= max_length) for t in texts]

    # extract each input_id arrays and convert to tensors
    return [torch.tensor(p['input_ids'], dtype=torch.long) for p in padded_titles]

####


def get_predicted_label_from_predictions(predictions):
    predicted_label = predictions.argmax(1).item()
    return predicted_label


####




def train(dataloader, model, optimizer, epoch: int, weights):
    """Run an epoch of training the model on the provided data, using the
    specified optimizer.
    """

    loss_fn = nn.NLLLoss(weight=weights)
    model.train()

    with tqdm(dataloader, unit="batch") as tbatch:
        for X, y in tbatch:

            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{CKPT_DIR}/ckpt_{epoch}.pt",
    )
   


def predict(data, model):
    predictions = []
    dataloader = DataLoader(data, batch_size=1)
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred)
    return predictions


def test(dataloader, model, dataset_name, weights):
    loss_fn = nn.NLLLoss(weight=weights)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"{dataset_name} Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>6f}\n"
    )


####


def make_or_restore_model(nfeat):
    # Either restore the latest model, or create a fresh one
    model = NN(nfeat).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    checkpoints = [
        CKPT_DIR + "/" + name
        for name in os.listdir(CKPT_DIR)
        if name[-1] == "t"
    ]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print("Restoring from", latest_checkpoint)
        ckpt = torch.load(latest_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt["epoch"]
        return model, optimizer, epoch + 1
    else:
        print("Creating a new model")
        return model, optimizer, 0


####



def precision(labels, model):
    true_pos = 0
    false_pos = 0

    print(len(model))

    for i in range(len(model)):
        label = labels[i]
        
        #print(label)
        #print(get_predicted_label_from_predictions(model[i]))

        if label == get_predicted_label_from_predictions(model[i]):
            if (get_predicted_label_from_predictions(model[i]) == 1):
                true_pos += 1
            
            else:
                false_pos += 1

    print(f"true pos: {true_pos}")

    return true_pos / (true_pos + false_pos)



def recall(labels, model):
    true_pos = 0
    false_neg = 0

    for i in range(len(model)):
        label = labels[i]
        
        if label == get_predicted_label_from_predictions(model[i]):
            if (get_predicted_label_from_predictions(model[i]) == 1):
                true_pos += 1
            
        else:
            if label != get_predicted_label_from_predictions(model[i]):
                false_neg += 1

    return true_pos / (true_pos + false_neg)



def f1_score(labels, model):

    #print(labels)
    #print([get_predicted_label_from_predictions(classification) for classification in model])

    p = precision(labels, model)
    r =  recall(labels, model)

    if (p + r == 0):
        return "Undefined"

    return (2 * p * r)/(p + r)



def frequency(label, labels):

    total_freq = 0

    for x in labels:

        if label == x:
            total_freq += 1

    return total_freq


#Fake-News-Phrase-Detection

def main():
    """Run the song classification."""

    csv.field_size_limit(sys.maxsize)

    train_f = "./data/train.csv"
    test_f = "./data/test.csv"

    train_titles, train_texts, train_labels = make_data(train_f)
    test_titles, test_texts, test_labels = make_data(test_f)

    print(test_labels)


    # for i in label_map_rev:
    #     print(f"Lyrics in Class {i} ({label_map_rev[i] + '):':14}",
    #           len([t for t in train_labels if t == i]))

    print()
    

    weight_map = {0: 1/(frequency(0, train_labels)), 1: 1/(frequency(1, train_labels))}
    tensor_weight_map = torch.tensor(list(weight_map.values()))

    train_feats_titles = prep_bert_data(train_titles, MAX_LENGTH)
    test_feats_titles = prep_bert_data(test_titles, MAX_LENGTH)

    train_dataset = list(zip(train_feats_titles, train_labels))
    test_dataset = list(zip(test_feats_titles, test_labels))

    print(train_labels)
    print(test_labels)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    #print(train_dataloader.dataset)

    model, optimizer, epoch_start = make_or_restore_model(MAX_LENGTH)
    

    for e in range(epoch_start, EPOCHS):
        print()
        print("Epoch", e)
        print("-------")

        model.train()
        train(train_dataloader, model, optimizer, e, tensor_weight_map)
    
        print()
    
        model.eval()
        test(train_dataloader, model, "Train", tensor_weight_map)
        test(test_dataloader, model, "Test", tensor_weight_map)
    
    test_predictions = predict(test_feats_titles, model)
    print(f"F1 Score for test data: {(f1_score(test_labels, test_predictions))}")
    #print(f"F1 Score for test data: {(f1_)}")
    print()


if __name__ == "__main__":
    main()
