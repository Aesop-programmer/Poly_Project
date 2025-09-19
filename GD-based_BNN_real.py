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
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# === 2) Model ===
model = models.__dict__["resnet"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model = model(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# --------------------
# 定義 Attribution 方法
# --------------------
# Gradient (Saliency)
saliency = Saliency(model)


# Integrated Gradients
ig = IntegratedGradients(model)


# SmoothGrad (基於 Saliency)
smoothgrad = NoiseTunnel(saliency)


# GradCAM (需要指定 layer)
# ResNet50 的最後一層卷積
# 可以用 list(model.children()) 找到正確的 layer
layer_gc = LayerGradCam(model, list(model.children())[-3][-1].conv2)


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

def normalize_attr(attr):
    attr = attr.squeeze().cpu().detach().numpy()
    # min-max normalization 到 [0,1]
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    if attr.ndim == 3:
        attr = np.transpose(attr, (1, 2, 0))
    return attr

def clamp_attr(attr):
    attr = attr.squeeze().cpu().detach().numpy()
    threshold = np.percentile(np.abs(attr), 95)
    attr = np.clip(attr, -threshold, threshold)
    
    return attr

def generate_diiferent_examples(model, testset, type):
    model.to("cuda")
    match type:
        case "best5":
            scores = []
            for i in range(len(testset)):
                x = testset[i][0].unsqueeze(0).to("cuda")
                y = testset[i][1]
                with torch.no_grad():
                    scores.append(model(x).max(1)[0].item())
            indices = [scores.index(x) for x in sorted(scores, reverse=True)[:5]]
        case "worst5":
            scores = []
            for i in range(len(testset)):
                x = testset[i][0].unsqueeze(0).to("cuda")
                y = testset[i][1]
                with torch.no_grad():
                    scores.append(model(x).max(1)[0].item())
            indices = [scores.index(x) for x in sorted(scores)[:5]]
        case "random5":
            import random
            indices = random.sample(range(0, len(testset)), 5)
    x = []
    y = []
    for i in indices:
        x.append(testset[i][0])
        y.append(testset[i][1])
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    
    images, labels = x.cuda(), y.cuda()
    preds = model(images).argmax(dim=1)



    plt.figure(figsize=(15, 24))

    vis_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    visset = datasets.CIFAR10(root='./data', train=False, transform=vis_transform)
    vis_x = []
    vis_y = []
    for i in indices:
        vis_x.append(visset[i][0])
        vis_y.append(visset[i][1])
    import numpy as np
    vis_x = np.array(vis_x)
    vis_y = np.array(vis_y)
    vis_x = torch.from_numpy(vis_x)
    vis_y = torch.from_numpy(vis_y)


    for idx in range(5):
        x = images[idx].unsqueeze(0).requires_grad_()
        y = labels[idx].item()
        pred = preds[idx].item()


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
        
        # 標題 (顯示正確與否)
        gt_name = CIFAR10_CLASSES[y]
        pred_name = CIFAR10_CLASSES[pred]
        correct = "Correct" if y == pred else "Wrong"
        title = f"GT: {gt_name} | Pred: {pred_name} {correct}"


        # 畫圖 (5 張)
        plt.subplot(5, 5, idx * 5 + 1)
        plt.imshow(tensor_to_img(vis_x[idx].cpu()))
        plt.title(title, fontsize=15)
        plt.axis("off")


        plt.subplot(5, 5, idx * 5 + 2)
        attr_grad = torch.tensor(attr_grad)
        attr_grad = attr_grad.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_grad), cmap="gray")
        plt.title("Gradient", fontsize=15)
        plt.axis("off")

        plt.subplot(5, 5, idx * 5 + 3)
        attr_ig = torch.tensor(attr_ig)
        attr_ig = attr_ig.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_ig), cmap="gray")
        plt.title("Integrated Gradients", fontsize=15)
        plt.axis("off")


        plt.subplot(5, 5, idx * 5 + 4)
        attr_sg = torch.tensor(attr_sg)
        attr_sg = attr_sg.abs().max(dim=0)[0]
        plt.imshow(tensor_to_img(attr_sg), cmap="gray")
        plt.title("SmoothGrad", fontsize=15)
        plt.axis("off")


        plt.subplot(5, 5, idx * 5 + 5)
        attr_gc = torch.tensor(attr_gc)
        plt.imshow(tensor_to_img(vis_x[idx].cpu()))
        plt.imshow(tensor_to_img(attr_gc), cmap="seismic",alpha=0.5)
        plt.title("GradCAM", fontsize=15)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"attribution_results_{type}.png")
generate_diiferent_examples(model, testset, "best5")
generate_diiferent_examples(model, testset, "worst5")
generate_diiferent_examples(model, testset, "random5")