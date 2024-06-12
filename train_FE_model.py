import torch
import torch.nn as nn


from torch.utils.data import random_split, DataLoader, Dataset


import torchvision.models as models

import pandas as pd


import matplotlib.pyplot as plt


from PIL import Image


import cv2


import argparse
import os
import time


import copy
import yaml


import torchvision.transforms as transforms


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


class InceptionV3Regressor(nn.Module):

    def __init__(self):

        super(InceptionV3Regressor, self).__init__()

        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )

        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1)
        )

    def forward(self, x):

        x = self.inception(x)

        return x


class ImageQualityDataset(Dataset):

    def __init__(self, images_path, csv_path):

        self.images_path = images_path

        self.image_names = os.listdir(images_path)

        self.labels = pd.read_csv(csv_path)

        self.transform = transforms.Compose(
            [
                transforms.Resize((338, 338)),
                transforms.CenterCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):

        image_path = os.path.join(self.images_path, self.image_names[idx])

        label = self.labels.loc[
            self.labels["image"] == self.image_names[idx], "label"
        ].values[0]

        image = Image.open(image_path)
        image = self.transform(image)
        return image, label


def draw_plot(train_loss, val_loss):

    plt.plot(train_loss, "r", label="Training Loss")

    plt.plot(val_loss, "b", label="Validation Loss")

    plt.title("Model Loss")

    plt.ylabel("Loss")

    plt.xlabel("Epoch")

    plt.legend(["Train", "Val"], loc="upper right")

    plt.show()


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):

    print("Training model")
    since = time.time()

    train_loss_history = []

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 1000.0

    for epoch in range(num_epochs):

        print("-" * 10)

        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase

        for phase in ["train", "val"]:

            if phase == "train":

                model.train()  # Set model to training mode

            else:

                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            running_corrects = 0

            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)

                # zero the parameter gradients

                optimizer.zero_grad()

                # forward

                with torch.set_grad_enabled(phase == "train"):

                    if phase == "train":

                        outputs, aux_outputs = model(inputs)

                        loss = criterion(torch.flatten(outputs), labels.float())

                    else:

                        outputs = model(inputs)

                        loss = criterion(torch.flatten(outputs), labels.float())

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":

                        loss.backward()

                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model

            if phase == "val" and epoch_loss < best_loss:

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "val":

                val_loss_history.append(epoch_loss)

            else:

                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("Best val oss: {:4f}".format(best_loss))

    draw_plot(train_loss_history, val_loss_history)

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FE model")

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the directory containing dataset images",
    )

    parser.add_argument(
        "--csv_path", type=str, help="Path to the CSV file containing labels"
    )

    args = parser.parse_args()

    config = load_config("train_FE_model_config.yaml")

    image_dataset = ImageQualityDataset(args.dataset_path, args.csv_path)

    dataset_length = len(image_dataset)

    test_length = int((1 - config["train_size"]) * dataset_length)

    train_length = dataset_length - test_length
    train_dataset, test_dataset = random_split(
        image_dataset, [train_length, test_length]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\033[31m\nDevice: {}\n\033[0m".format(device))

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = {"train": train_loader, "val": test_loader}

    model = InceptionV3Regressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    trained_model = train_model(
        model, dataloaders, criterion, optimizer, device, num_epochs=config["epochs"]
    )

    torch.save(trained_model.state_dict(), "trained_model.pth")
