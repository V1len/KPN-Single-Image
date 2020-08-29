import time
import datetime
import os
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import argparse
import cv2
import torchvision.models as models
from torchvision import datasets, transforms
import scipy
import json
import csv
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class GANTrainDataset(Dataset):
    def __init__(self, transform=None):
        super(GANTrainDataset, self).__init__()
        self.base_root="/mnt/nvme/yihao/GANFingerprints/KPN_datasets/"
        GANs=["ProGAN","SNGAN","CramerGAN","MMDGAN"]
        self.transform=transform
        self.imglist=[]
        self.labellist=[]
        ind=1
        batch=2000
        #label 1 fake, label 0 real
        for i in GANs:
            GAN_root=self.base_root+i+"_celeba_align_png_cropped"
            count=0
            for l in os.listdir(GAN_root)[batch*(ind-1):batch*ind]:
                self.imglist.append(self.base_root+i+"_celeba_align_png_cropped/"+l)
            self.labellist=self.labellist+[1]*batch
            ind+=1
        real_image_path=self.base_root+"celeba_align_png_cropped_128"
        for l in os.listdir(real_image_path)[0:batch*4]:
            self.imglist.append(self.base_root+"celeba_align_png_cropped_128/"+l)
        self.labellist=self.labellist+[0]*batch*4
        print(len(self.imglist))
        print(len(self.labellist))


    def __getitem__(self, index):
        image_path = self.imglist[index]
        label = self.labellist[index]

        img = cv2.imread(image_path)
        # print(img.shape)
        # print(noisy_img.shape)

        if self.transform:
            img=self.transform(img)

        return img,label

    def __len__(self):
        return len(self.imglist)


class GANTestDataset(Dataset):
    def __init__(self, transform=None):
        super(GANTestDataset, self).__init__()
        self.base_root="/mnt/nvme/yihao/GANFingerprints/KPN_datasets/"
        GANs=["ProGAN","SNGAN","CramerGAN","MMDGAN"]
        self.transform=transform
        self.imglist=[]
        self.labellist=[]
        ind=1
        base=9000
        batch=200
        #label 1 fake, label 0 real
        for i in GANs:
            GAN_root=self.base_root+i+"_celeba_align_png_cropped"
            for l in os.listdir(GAN_root)[base+batch*(ind-1):base+batch*ind]:
                self.imglist.append(self.base_root+i+"_celeba_align_png_cropped/"+l)
            self.labellist=self.labellist+[1]*batch
            ind+=1
        real_image_path=self.base_root+"celeba_align_png_cropped_128"
        for l in os.listdir(real_image_path)[8000:8800]:
            self.imglist.append(self.base_root+"celeba_align_png_cropped_128/"+l)
        self.labellist=self.labellist+[0]*800
        print(len(self.imglist))
        print(len(self.labellist))


    def __getitem__(self, index):
        image_path = self.imglist[index]
        label = self.labellist[index]

        img = cv2.imread(image_path)
        # print(img.shape)
        # print(noisy_img.shape)

        if self.transform:
            img=self.transform(img)

        return img,label

    def __len__(self):
        return len(self.imglist)


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_each_epoch = []
    best_loss=5017600.0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_classification_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    classification_outputs = model(inputs)
                    classification_loss = criterion(classification_outputs, labels)
                    # print(classification_outputs)
                    # print(labels)
                    # print(torch.max(classification_outputs,axis=1))
                    preds = torch.max(classification_outputs,axis=1)[1]
                    loss=classification_loss

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                #loss and corrects of per batch
                running_loss += loss.item()
                current_batch_classification_corrects=torch.sum(preds == labels.data).double()
                running_classification_corrects+=current_batch_classification_corrects
                #print out the loss and corrects of per img
                print('{} Epoch Loss: {:.4f} C Accuracy:{:.4f}'.format(phase,loss.item()/batch_size,current_batch_classification_corrects/batch_size))


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_classification_acc = running_classification_corrects / len(dataloaders[phase].dataset)
            loss_each_epoch.append(epoch_loss)

            print('{} Loss: {:.4f} C Acc: {:.4f}'.format(phase, epoch_loss, epoch_classification_acc))
            if phase == 'val':
                epoch_end_time = time.time() - epoch_start_time
                print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch, epoch_end_time // 60, epoch_end_time % 60))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Loss of each epoch: ')
    for i, l in enumerate(loss_each_epoch):
        print(i," ",l)

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    torch.save(model,'resnet50_model_for_GANF.pkl')
    return model, val_acc_history


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.fc=nn.Linear(in_features=2048, out_features=2)
    model=model.cuda().eval()

    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = GANTrainDataset(transform=data_transform)
    test_dataset = GANTestDataset(transform=data_transform)

    batch_size=64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = test_loader

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,betas=(0.9, 0.999), weight_decay=1e-7)

    best_model,history=train_model(model, dataloaders,criterion,optimizer,10)

