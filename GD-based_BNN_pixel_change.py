# pip install torch torchvision captum pytorch-grad-cam

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import LayerGradCam
import models

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
# === 1) Data ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**__imagenet_stats)
])



# === 2) Model ===
model_bin = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_bin = model_bin(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar")
model_bin.load_state_dict(checkpoint['state_dict'])
model_bin.eval()

model_real = models.__dict__["resnet"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_real = model_real(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar")
model_real.load_state_dict(checkpoint['state_dict'])
model_real.eval()


import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# === 1) CIFAR-100 dataset (只做 ToTensor，不做 Normalize) ===
cifar100_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())


# === 2) 原始影像 → Normalize 再進模型 ===
def normalize(img):
    mean = torch.tensor(__imagenet_stats['mean']).view(3,1,1)
    std = torch.tensor(__imagenet_stats['std']).view(3,1,1)
    return (img - mean) / std

def denormalize(img):
    mean = torch.tensor(__imagenet_stats['mean']).view(3,1,1)
    std = torch.tensor(__imagenet_stats['std']).view(3,1,1)
    return img * std + mean

CIFAR10_CLASSES = [
"airplane", # 0
"automobile",# 1
"bird", # 2
"cat", # 3
"deer", # 4
"dog", # 5
"frog", # 6
"horse", # 7
"ship", # 8
"truck" # 9
]

import numpy as np
# === Function: 測試單張影像 (batch 版本) ===
def test_image_batch(img_raw, model, values_to_test=[0.0, 0.25 ,0.5, 0.75, 1.0], batch_size=512):
    device = next(model.parameters()).device

    # baseline prediction
    x = normalize(img_raw).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_orig = model(x).argmax(dim=1).item()

    # 產生所有修改版本
    imgs_mod = []
    locs = []   # 記錄 (ch,row,col)
    for ch in range(3):
        for row in range(32):
            for col in range(32):
                orig_val = img_raw[ch, row, col].item()
                for v in values_to_test:
                    if abs(v - orig_val) < 1e-6:
                        continue
                    img_mod = img_raw.clone()
                    img_mod[ch, row, col] = v
                    imgs_mod.append(normalize(img_mod))
                    locs.append((ch, row, col))

    imgs_mod = torch.stack(imgs_mod).to(device)  # (N, 3, 32, 32)

    # 分批 forward
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs_mod), batch_size):
            batch = imgs_mod[i:i+batch_size]
            out = model(batch).argmax(dim=1).cpu()
            preds.append(out)
    preds = torch.cat(preds).numpy()

    # 判斷哪些 pixel 有改變
    change_mask = torch.zeros((32,32))
    statistics = {}
    for i in range(10):
        statistics[i] = 0
    example =[]
    current = 0
    for (ch,row,col), p in zip(locs, preds):
        statistics[p] += 1
        if p != pred_orig:
            change_mask[row, col] = 1
            example.append((p, imgs_mod[current]))
        current += 1
    total_pixels = img_raw.numel()
    changed_pixels = (change_mask==1).sum().item()
    ratio = changed_pixels / (32*32)
    return pred_orig, ratio, change_mask, statistics, example



import random
indices = random.sample(range(0, len(cifar100_raw)), 5)
def generate_result(model, testset, indices,type):
    results = []
    for i in indices:
        while True:
            img_raw, label = testset[i]
            pred_orig, ratio, change_mask, statistics, example = test_image_batch(img_raw, model)
            if len(example) == 0:
                i = random.sample(range(0, len(cifar100_raw)), 1)[0]
                pass
            else:

                results.append((img_raw, pred_orig, label, change_mask, statistics, example))
                break
    

    # === 視覺化幾張 sample ===
    num_show = len(results)
    plt.figure(figsize=(15, 24))
    
    for idx in range(num_show):
        img_raw, pred_orig, label, change_mask, statistics, example = results[idx]
        
        plt.subplot(5, 5, idx * 5 + 1)
        plt.imshow(img_raw.permute(1,2,0).numpy())
        plt.title(f"Image: Pred={CIFAR10_CLASSES[pred_orig]}, True={CIFAR10_CLASSES[label]}")
        plt.axis("off")

        mask_rgb = np.zeros((32,32,3))
        mask_rgb[...,0] = change_mask.numpy()
        overlay = 0.7*img_raw.permute(1,2,0).numpy() + 0.3*mask_rgb
        plt.subplot(5, 5, idx * 5 + 2)
        plt.imshow(overlay)
        rank = sorted(statistics.items(), key=lambda x: x[1], reverse=True)[:3]
        if rank[1][1] == 0 and rank[2][1] == 0: 
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}")
        elif rank[2][1] == 0:
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}, {CIFAR10_CLASSES[rank[1][0]]}")
        else:
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}, {CIFAR10_CLASSES[rank[1][0]]}, {CIFAR10_CLASSES[rank[2][0]]}")
        plt.axis("off")
        
        random.shuffle(example)
        for j in range(min(3, len(example))):
            cls, img_mod = example[j]
            plt.subplot(5, 5, idx * 5 + 3 + j)
            plt.imshow(denormalize(img_mod).permute(1,2,0).cpu().numpy())
            plt.title(f"Pred={CIFAR10_CLASSES[cls]}")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"pixel_change_analysis_{type}.png")


# generate_result(model_real, cifar100_raw, indices, "real")
# generate_result(model_bin, cifar100_raw, indices, "binary_wo_bn")
generate_result(model_bin, cifar100_raw, indices, "binary_w_bn")