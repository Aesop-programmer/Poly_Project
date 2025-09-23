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
import matplotlib.pyplot as plt
import numpy as np
# --------------------
# 視覺化
# --------------------
def tensor_to_img(t):
    """把 (C,H,W) tensor 轉成 numpy image"""
    t = t.squeeze().cpu().detach().numpy()
    if t.ndim == 3:
        t = np.transpose(t, (1, 2, 0))
    return t

def normalize_attr(attr):
    attr = attr.squeeze().cpu().detach().numpy()
    # min-max normalization 到 [0,1]
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    if attr.ndim == 3:
        attr = np.transpose(attr, (1, 2, 0))
    return attr

def clamp_attr(attr):
    attr = attr.squeeze().cpu().detach().numpy()
    threshold = np.percentile(np.abs(attr), 99)
    attr = np.clip(attr, -threshold, threshold)
    
    return attr

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
            
    from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
    from captum.attr import visualization as viz
    from captum.attr import LayerGradCam

    
    saliency = Saliency(model)


    # Integrated Gradients
    ig = IntegratedGradients(model)


    # SmoothGrad (基於 Saliency)
    smoothgrad = NoiseTunnel(saliency)


    # 可以用 list(model.children()) 找到正確的 layer
    if type == "real":
        layer_gc = LayerGradCam(model, list(model.children())[-3][-1].conv2)
    else:
        layer_gc = LayerGradCam(model, list(model.children())[-6][-1].conv2)
        
    # === 視覺化幾張 sample ===
    num_show = len(results)
    plt.figure(figsize=(15, 24))
    
    for idx in range(num_show):
        img_raw, pred_orig, label, change_mask, statistics, example = results[idx]
        
        plt.subplot(5, 6, idx * 6 + 1)
        plt.imshow(img_raw.permute(1,2,0).numpy())
        plt.title(f"Image: Pred={CIFAR10_CLASSES[pred_orig]}, True={CIFAR10_CLASSES[label]}")
        plt.axis("off")

        mask_rgb = np.zeros((32,32,3))
        mask_rgb[...,0] = change_mask.numpy()
        overlay = 0.7*img_raw.permute(1,2,0).numpy() + 0.3*mask_rgb
        plt.subplot(5, 6, idx * 6 + 2)
        plt.imshow(overlay)
        rank = sorted(statistics.items(), key=lambda x: x[1], reverse=True)[:3]
        if rank[1][1] == 0 and rank[2][1] == 0: 
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}")
        elif rank[2][1] == 0:
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}, {CIFAR10_CLASSES[rank[1][0]]}")
        else:
            plt.title(f"top3: {CIFAR10_CLASSES[rank[0][0]]}, {CIFAR10_CLASSES[rank[1][0]]}, {CIFAR10_CLASSES[rank[2][0]]}")
        plt.axis("off")
        
        x = normalize(img_raw).unsqueeze(0).requires_grad_()
        y = label
        pred = pred_orig

        # Attribution
        attr_grad = saliency.attribute(x, target=pred)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        mean_img = mean.expand(1, 3, 32, 32).to("cuda")
        attr_ig = ig.attribute(x, target=y, n_steps=300)
        attr_sg = smoothgrad.attribute(x, nt_type="smoothgrad", nt_samples=500, stdevs=0.1, target=pred)
        attr_gc = layer_gc.attribute(x, target=pred)
        attr_gc = F.interpolate(attr_gc, size=(32, 32), mode='bilinear', align_corners=False)

        attr_grad = clamp_attr(attr_grad)
        attr_ig = clamp_attr(attr_ig)
        attr_sg = clamp_attr(attr_sg)
        attr_gc = clamp_attr(attr_gc)
        
        plt.subplot(5, 6, idx * 6 + 3)
        attr_grad = torch.tensor(attr_grad)
        attr_grad = attr_grad.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_grad), cmap="gray")
        plt.title("Gradient", fontsize=15)
        plt.axis("off")

        plt.subplot(5, 6, idx * 6 + 4)
        attr_ig = torch.tensor(attr_ig)
        attr_ig = attr_ig.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_ig), cmap="gray")
        plt.title("Integrated Gradients", fontsize=15)
        plt.axis("off")


        plt.subplot(5, 6, idx * 6 + 5)
        attr_sg = torch.tensor(attr_sg)
        attr_sg = attr_sg.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_sg), cmap="gray")
        plt.title("SmoothGrad", fontsize=15)
        plt.axis("off")


        plt.subplot(5, 6, idx * 6 + 6)
        attr_gc = torch.tensor(attr_gc)
        plt.imshow(img_raw.permute(1,2,0).numpy())
        plt.imshow(tensor_to_img(attr_gc), cmap="seismic",alpha=0.5)
        plt.title("GradCAM", fontsize=15)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(f"pixel_change_vs_GD_{type}.png")


# generate_result(model_real, cifar100_raw, indices, "real")
# generate_result(model_bin, cifar100_raw, indices, "binary_wo_bn")
generate_result(model_bin, cifar100_raw, indices, "binary_w_bn")