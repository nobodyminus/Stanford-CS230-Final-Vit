import argparse
import glob
import os
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

from models.attentions import AttentionSelector
from models.vit import ViT
from datasets import load_dataset
import sys


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


class ImagenetFromHuggingFace(Dataset):
    def __init__(self, hfds, transform=None):
        self.hfds = hfds
        self.transform = transform

    def __len__(self):
        return len(self.hfds)

    def __getitem__(self, idx):
        item = self.hfds[idx]
        img = item['image']
        img_transformed = self.transform(img)
        if img_transformed.shape[0] == 1:
            img_transformed = img_transformed.repeat(3, 1, 1).squeeze()

        label = item['label']

        return img_transformed, label


def get_dataloaders(batch_size):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_data = load_dataset('Maysee/tiny-imagenet', split='train')
    valid_data = load_dataset('Maysee/tiny-imagenet', split='valid')
    train_data = ImagenetFromHuggingFace(train_data, train_transforms)
    valid_data = ImagenetFromHuggingFace(valid_data, val_transforms)
    #train_data = torch.utils.data.Subset(train_data, list(range(0, len(train_data)//20)))
    #valid_data = torch.utils.data.Subset(valid_data, list(range(0, len(valid_data)//20)))
    train_length = len(train_data)
    train_indices = list(range(0, int(train_length*0.95)))
    test_indices = list(range(int(train_length * 0.95)+1, train_length))
    test_data = torch.utils.data.Subset(train_data, test_indices)
    train_data = torch.utils.data.Subset(train_data, train_indices)
    train_loader_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    test_loader_ = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_loader_, valid_loader_, test_loader_


def train(model, train_loader_, valid_loader_, test_loader_, device, lr, epochs):
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader_):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader_)
            epoch_loss += loss / len(train_loader_)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader_:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader_)
                epoch_val_loss += val_loss / len(valid_loader_)
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}"
            f" - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    with torch.no_grad():
        test_accuracy = 0
        test_loss = 0
        for data, label in test_loader_:
            data = data.to(device)
            label = label.to(device)

            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            test_accuracy += acc / len(test_loader_)
            test_loss += val_loss / len(test_loader_)

    print(
        f"Test - loss : {test_loss:.4f} - acc: {test_accuracy:.4f}:"
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda', help='What device to use')
    parser.add_argument('-b', '--batch_size', default=24)
    parser.add_argument('-e', '--epochs', default=20)
    parser.add_argument('-lr', '--lr', default=3e-5)
    parser.add_argument('-g', '--gamma', default=0.7)
    parser.add_argument('-a', '--attention', default='BASELINE', help='Attention type. Options are BASELINE for normal'
                                                                      'transformer, MAX for the pointwise maximum,'
                                                                      'and MESSAGE for message passing')
    args = parser.parse_args()
    sys.stdout = open(f'{args.attention}.txt', 'w')
    vit = ViT(
        dim=256,
        image_size=224,
        patch_size=32,
        num_classes=200,
        channels=3,
        depth=12,
        heads=12,
        mlp_dim=512,
        attention=AttentionSelector[args.attention].value).to(args.device)
    #max_vit = ViT(
    #    dim=256,
    #    image_size=224,
    #    patch_size=32,
    #    num_classes=200,
    #    channels=3,
    #    depth=12,
    #    heads=12,
    #    mlp_dim=512,
    #    attention=AttentionSelector['MAX'].value).to(args.device)
    #message_vit = ViT(
    #    dim=256,
    #    image_size=224,
    #    patch_size=32,
    #    num_classes=200,
    #    channels=3,
    #    depth=12,
    #    heads=12,
    #    mlp_dim=512,
    #    attention=AttentionSelector['MESSAGE'].value).to(args.device)

    train_loader, valid_loader, test_loader = get_dataloaders(args.batch_size)
    print(args.attention)
    train(vit, train_loader, valid_loader, test_loader, args.device, args.lr, args.epochs)
    #print('MAX')
    #train(max_vit, train_loader, valid_loader, test_loader, args.device, args.lr, args.epochs)
    #print('BASELINE')
    #train(baseline_vit, train_loader, valid_loader, test_loader, args.device, args.lr, args.epochs)
