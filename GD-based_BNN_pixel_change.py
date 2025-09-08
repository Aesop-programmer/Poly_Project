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
model = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model = model(**model_config)
checkpoint = torch.load("/home/aesop/BNN/Gradient_based/BinaryNet.pytorch/results/2025-08-30_03-10-40/model_best.pth.tar")
# model = models.__dict__["resnet"]
# model_config = {'input_size': 32, 'dataset': "cifar10"}
# model = model(**model_config)
# checkpoint = torch.load("/home/aesop/BNN/Gradient_based/BinaryNet.pytorch/results/2025-08-30_03-13-26/model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()


import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# === 1) CIFAR-100 dataset (只做 ToTensor，不做 Normalize) ===
cifar100_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

# 取一張影像
img_raw, label = cifar100_raw[0]   # shape: (3, 32, 32), 範圍 [0,1]

# === 2) 原始影像 → Normalize 再進模型 ===
def normalize(img):
    mean = torch.tensor(__imagenet_stats['mean']).view(3,1,1)
    std = torch.tensor(__imagenet_stats['std']).view(3,1,1)
    return (img - mean) / std

x = normalize(img_raw).unsqueeze(0)

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
    for (ch,row,col), p in zip(locs, preds):
        if p != pred_orig:
            change_mask[row, col] = 1

    total_pixels = img_raw.numel()
    changed_pixels = (change_mask==1).sum().item()
    ratio = changed_pixels / (32*32)
    return pred_orig, ratio, change_mask

# === 多張圖一起跑 ===
num_images = 10
ratios = []
results = []

for i in range(num_images):
    img_raw, label = cifar100_raw[i]
    pred_orig, ratio, change_mask = test_image_batch(img_raw, model)
    ratios.append(ratio)
    results.append((img_raw, pred_orig, label, change_mask))
    print(f"Image {i}: True={label}, Pred={pred_orig}, Changed ratio={ratio:.2%}")

print(f"\nAverage changed ratio over {num_images} images: {np.mean(ratios):.2%}")

# === 視覺化幾張 sample ===
num_show = min(10, num_images)
fig, axs = plt.subplots(num_show, 2, figsize=(8, 3*num_show))

for idx in range(num_show):
    img_raw, pred_orig, label, change_mask = results[idx]
    axs[idx,0].imshow(img_raw.permute(1,2,0).numpy())
    axs[idx,0].set_title(f"Image {idx}: Pred={pred_orig}, True={label}")
    axs[idx,0].axis("off")

    mask_rgb = np.zeros((32,32,3))
    mask_rgb[...,0] = change_mask.numpy()
    overlay = 0.7*img_raw.permute(1,2,0).numpy() + 0.3*mask_rgb
    axs[idx,1].imshow(overlay)
    axs[idx,1].set_title(f"Changed pixels (red), ratio={ratios[idx]:.2%}")
    axs[idx,1].axis("off")

plt.savefig("pixel_change_analysis.png")