#!/usr/bin/env python3

# Assignment 5
# CMPU 366, Spring 2024

import csv
import os
import sys
from typing import Callable, Tuple

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
MAX_LENGTH = 15
LR = 0.001
CKPT_DIR = "./ckpt"
NUM_CLASSES = 2

N_DIMS = 768


label_map = {"Real": 0, "Fake": 1}
label_map_rev = {0: "Real", 1: "Fake"}


class NN(nn.Module):
    def __init__(self, n_features: int):
        """Construct the pieces of the neural network."""

        # Initialize the parent class (nn.Module)
        super(NN, self).__init__()
        
        self.bert = bert

        self.hidden_layers = nn.Sequential(      

                            # Layer 1
                            nn.Linear(N_DIMS * n_features, N_DIMS * n_features // 2),
                            nn.ReLU(),

                            # Layer 2
                            nn.Linear(N_DIMS * n_features // 2, N_DIMS * n_features // 2),
                            nn.ReLU(),

                            # Layer 3
                            nn.Linear(N_DIMS * n_features // 2, N_DIMS * n_features // 4),
                            nn.ReLU(),

                            # Layer 4
                            nn.Linear(N_DIMS * n_features // 4, N_DIMS * n_features // 8),
                            nn.ReLU(),

                            # Layer 5
                            nn.Linear(N_DIMS * n_features // 8, N_DIMS * n_features // 16),
                            nn.ReLU(),

                            # Layer 6
                            nn.Linear(N_DIMS * n_features // 16, N_DIMS * n_features // 16),
                            nn.ReLU(),

                            # Layer 7
                            nn.Linear(N_DIMS * n_features // 16, N_DIMS * n_features // 32),
                            nn.ReLU(),

                            # Layer 8
                            nn.Linear(N_DIMS * n_features // 32, N_DIMS * n_features // 32),
                            nn.ReLU(),
                            
                            # Layer 9
                            nn.Linear(N_DIMS * n_features // 32, N_DIMS * n_features // 64),
                            nn.ReLU(),
                            
        )
        self.flatten = nn.Flatten()
        # Reduce to the 2 labels (real or fake)
        # Layer 10

        self.output_layer = nn.Linear(N_DIMS * n_features // 64, 2)

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


def make_data(fname: str, label_map: dict) -> Tuple[list[str], list[str], list[int]]:
    with open(fname, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        
        saved = ([], [])

        for row in data:
            # Gets title
            saved[0].append(row[0])

            # Gets text
            saved[1].append(row[1])

        return saved




def prep_bert_data(titles: list[str], max_length: int) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    padded_titles = [tokenizer(t, truncation= True, padding='max_length', max_length= max_length) for t in titles]
    #padded_texts = [tokenizer(t, truncation= True, padding='max_length', max_length= max_length) for t in texts]
    # print(padded[:2])
    # extract each input_id arrays and convert to tensors
    return [torch.tensor(p['input_ids'], dtype=torch.long) for p in padded_titles]

####


def get_predicted_label_from_predictions(predictions):
    predicted_label = predictions.argmax(1).item()
    return predicted_label


def sample_and_print_predictions(feats, data, labels, model):
    import random
    prediction = predict(feats, model)
    print(prediction[:2])

    length = len(data)
    for i in range(10):
        j = random.randint(0, length)

        predicted_artist = get_predicted_label_from_predictions(prediction[i])
        
        print("  Lyrics: ", data[j])
        print("- Class: ", label_map_rev[labels[j]])
        print("- Prediction:", label_map_rev[predicted_artist])
        print()


####




def train(dataloader, model, optimizer, epoch: int):
    """Run an epoch of training the model on the provided data, using the
    specified optimizer.
    """
    
    # Get array of labels of authors and convert to a tensor
    authors = torch.tensor([data[1] for data in dataloader.dataset])
    
    # Get the frequency 
    weights = torch.bincount(authors)

    # Get inverse of frequency
    inversed = 1 / weights.float()
    
   

    loss_fn = nn.NLLLoss(weight=inversed)
    model.train()
    with tqdm(dataloader, unit="batch") as tbatch:
        for X, y in tbatch:
            X = X.to(device)
            y = y.to(device)

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


def test(dataloader, model, dataset_name):
    loss_fn = nn.NLLLoss()
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

def print_performance_by_class(labels, model):
    total = len(model)
    acc_by_cat = [[0,0], [0,0], [0,0]]
    # acc_by_cat = (correct, total)


    for i in range(total):
        label = labels[i]
        
        if label == get_predicted_label_from_predictions(model[i]) :
            # Increment correct by one
            acc_by_cat[label][0] += 1
        
        # Always Increment total for artist
        acc_by_cat[label][1] += 1
    
    
    print("Accuracy by Category:")
    i = 0
    for stat in acc_by_cat:
        accuracy = stat[0] / stat[1] 
        
        print(
            f"{label_map_rev[i]}: {(100 * accuracy):>0.1f}\n"
        )
        i+=1




#Fake-News-Phrase-Detection

def main():
    """Run the song classification."""

    train_f = "train.csv"
    test_f = "test.csv"

    train_titles, train_labels = make_data(train_f, label_map)
    test_titles, test_labels = make_data(test_f, label_map)

    # for i in label_map_rev:
    #     print(f"Lyrics in Class {i} ({label_map_rev[i] + '):':14}",
    #           len([t for t in train_labels if t == i]))

    # print()

    train_feats_titles = prep_bert_data(train_titles, MAX_LENGTH)
    test_feats_titles = prep_bert_data(test_titles, MAX_LENGTH)

    train_dataset = list(zip(train_feats_titles, train_labels))
    test_dataset = list(zip(test_feats_titles, test_labels))

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model, optimizer, epoch_start = make_or_restore_model(MAX_LENGTH)

    for e in range(epoch_start, EPOCHS):
        print()
        print("Epoch", e)
        print("-------")
    
        model.train()
        train(train_dataloader, model, optimizer, e)
    
        print()
    
        model.eval()
        test(train_dataloader, model, "Train")
        test(test_dataloader, model, "Test")
    
    test_predictions = predict(test_feats_titles, model)
    print_performance_by_class(test_labels, test_predictions)
    print()

    sample_and_print_predictions(test_feats_titles, test_titles, test_labels,model)


if __name__ == "__main__":
    main()
