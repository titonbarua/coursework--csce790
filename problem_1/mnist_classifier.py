# This module contains MNIST classifier code from
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
# with some modifications.


import sys
import argparse

import numpy as np
import torch
import random
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use('TkAgg')

INPUT_SIZE = 784
OUTPUT_SIZE = 10
SAVEPATH_FMT = "mnist_{hidden_sizes}.pt"

def download_mnist_data(
        normalization_mean=0.5,
        normalization_stddev=0.5,
        batch_size=64):
    # Create a transform to normalize the images with 0.5 mean and 0.5 std
    # deviation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (normalization_mean,), (normalization_stddev,))])

    # Download training set from official channel.
    trainset = datasets.MNIST(
        './mnist_trainset',
        download=True,
        train=True,
        transform=transform)

    # Download validation set from official channel.
    valset = datasets.MNIST(
        './mnist_testset',
        download=True,
        train=False,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True)

    return (trainloader, valloader)


def show_random_images(dataloader, row=4, col=3, title=None):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(
        row, col, figsize=(row * 1.2, col * 1.2))
    n_images = len(images)
    n_samples = row * col
    sample_indices = random.sample(range(1, n_images), n_samples)

    if title:
        fig.suptitle(title, fontsize='x-large')

    for i in range(row):
        for j in range(col):
            idx = sample_indices[i * col + j]
            axes[i, j].axis('off')
            axes[i, j].imshow(images[idx].numpy().squeeze(), cmap='gray_r')

    plt.show()


def create_model(
        hidden_sizes=[128, 64],
        device='cpu'):
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], OUTPUT_SIZE),
        nn.LogSoftmax(dim=1))

    model.to(device)
    return model


def logps_to_class(log_prob_scores):
    prob_scores = torch.exp(log_prob_scores)
    prob_scores = prob_scores.cpu().numpy()
    class_labels = np.argmax(prob_scores, axis=1)
    return class_labels


def train_model(
        model,
        trainloader,
        valloader,
        epochs=15,
        device='cpu'):
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images.to(device))  # Calculate log probabilities.
    loss = criterion(logps, labels.to(device))  # Calculate NLL loss.

    print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images.
            images = images.view(images.shape[0], -1)

            # Training pass.
            optimizer.zero_grad()

            output = model(images.cuda())
            loss = criterion(output, labels.cuda())

            # Calculate gradients with back-propagation.
            loss.backward()

            # Optimize the weights.
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}"
                  .format(e, running_loss/len(trainloader)))

    print("\nTraining Time (in minutes) = ", (time() - time0) / 60)


def evaluate_performance(
        model,
        valloader,
        confusion_matrix_savepath=None,
        device='cpu'):
    expected = []
    predicted = []

    with torch.no_grad():
        for test_images, expected_labels in valloader:
            test_images = test_images.view(test_images.shape[0], -1)
            predicted_labels = logps_to_class(
                model(test_images.to(device)))

            expected += expected_labels
            predicted += list(predicted_labels)

    assert len(expected) == len(predicted)

    expected = np.array(expected)
    predicted = np.array(predicted)

    # print(expected.shape)
    # print(predicted.shape)
    # print(predicted[:3])

    avg_accuracy = accuracy_score(expected, predicted)
    precision = precision_score(expected, predicted, average=None)
    recall = recall_score(expected, predicted, average=None)

    print(f"Average accuracy: {avg_accuracy}")
    print(f"Precision:")
    for label, val in zip(range(0, 10), precision):
        print(f"\tClass '{label}': {val:.3f}")
    print(f"Recall:")
    for label, val in zip(range(0, 10), recall):
        print(f"\tClass '{label}': {val:.3f}")

    ConfusionMatrixDisplay.from_predictions(expected, predicted)
    if confusion_matrix_savepath:
        plt.savefig(confusion_matrix_savepath, dpi=300)
    else:
        plt.show()


cmdparser = argparse.ArgumentParser("train_mnist.py")
cmdparser.add_argument("--no-train", action="store_true", default=False)
cmdparser.add_argument("--epochs", action="store", type=int, default=15)
cmdparser.add_argument("--hidden_sizes", action="store", default="128x64")

if __name__ == "__main__":
    args = cmdparser.parse_args()

    hidden_sizes_str = args.hidden_sizes
    savepath = SAVEPATH_FMT.format(
        hidden_sizes=hidden_sizes_str)
    hidden_sizes = [int(n) for n in hidden_sizes_str.split("x")]
    epochs = args.epochs

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    trainloader, valloader = download_mnist_data()
    show_random_images(trainloader, title="Random Training Samples")

    if args.no_train:
        print(f"Loading model from {savepath} ...")
        model = torch.load(savepath, weights_only=False)
        model.eval()
        model.to(device)
    else:
        model = create_model(hidden_sizes, device=device)
        train_model(
            model,
            trainloader=trainloader,
            valloader=valloader,
            epochs=epochs,
            device=device)
        print(f"Saving model to {savepath} ...")
        torch.save(model, savepath)

    evaluate_performance(
        model,
        valloader,
        confusion_matrix_savepath=f"confusion_matrix_{hidden_sizes_str}.png",
        device=device)
