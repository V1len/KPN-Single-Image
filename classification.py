import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import datetime
import time
import csv
from torchvision import models
import glob
import os, sys
import random
import shutil
from PIL import Image
import cv2
import numpy as np
import KPN


def WriteMapping(txt_root, csv_root):
    with open(csv_root, 'w', encoding='utf-8') as csv_f:
        with open(txt_root, encoding="UTF-8") as txt_f:
            for line in txt_f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                if(line != ""):
                    list = line.split(' ', 4)
                    id = list[0]
                    code = list[1]
                    csv_writer = csv.writer(csv_f)
                    csv_writer.writerow([id,code])
            txt_f.close()
        csv_f.close()

def ReadMapping(csv_root):
    with open(csv_root,"r",encoding='utf-8') as csv_f:

        csv_f.close()

def CopyImage(root, target):
    dirs = os.listdir(root)  # 列出所有图像的名称
    num = 5000
    sample = random.sample(dirs, num)
    print(sample)
    for name in sample:
        if os.path.exists(root + name):
            shutil.copy(root + name, target + name)
        else:
            continue

def Resize(root):
    path_to_images = root
    all_images = glob.glob(path_to_images + '*')
    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((224, 224), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 500 == 0:
            print(i)

def Save_Image(dir,image):
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = image * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
    image = image * 255.0
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dir, image)

def Save_NoiseAndDenoise_Image(root_dir, noise_dir, denoise_dir, load_model_dir):
    net = KPN.KPN().cuda()
    net.load_state_dict(torch.load(load_model_dir))
    net.eval()
    data_transforms = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    images = os.listdir(root_dir)
    for i in range(len(images)):
        img_path = os.path.join(root_dir, images[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = data_transforms(img)

        noise_img = np.random.normal(0, 30, img.shape).astype(np.float32)
        noise_img = noise_img / 255.0
        noisy_img = img + noise_img
        save_noise_path = os.path.join(noise_dir, images[i])
        Save_Image(save_noise_path, noisy_img)

        true_input = noisy_img.unsqueeze(0).cuda()
        output = net(true_input, true_input)
        denoise_img = output.squeeze(0)
        save_denoise_path = os.path.join(denoise_dir, images[i])
        Save_Image(save_denoise_path, denoise_img)

def Valiation(net, class_map_dir):
    class_map = {}
    with open(class_map_dir, 'r') as csv_f:
        reader = csv.reader(csv_f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            id = row[0]
            code = row[1]
            class_map[code] = id



if __name__ == '__main__':
    # WriteMapping("C:/Users/lab-301/Desktop/1.txt", "C:/Users/lab-301/Desktop/classMap.csv")

    # rootDir = "D:/mini_ImageNet/images/"
    # tarDir = "D:/PycharmProjects/proc_images/samples/"
    # CopyImage(rootDir,tarDir)
    # Resize(tarDir)

    image_gt_dir = "./image_gt"
    image_noisy_dir = "./image_noisy"
    image_pred_dir = "./image_pred"
    # load_model_dir = "./KPNmodels/KPN_epoch20.pth"
    # Save_NoiseAndDenoise_Image(image_gt_dir, image_noisy_dir, image_pred_dir, load_model_dir)

    net = models.resnet50(pretrained=True)
    class_map_dir = "./classMap.csv"
    Valiation(net, class_map_dir)

