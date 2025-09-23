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
model_bin_w_bn = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_bin_w_bn = model_bin_w_bn(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar")
model_bin_w_bn.load_state_dict(checkpoint['state_dict'])
model_bin_w_bn.eval()

model_bin_wo_bn = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_bin_wo_bn = model_bin_wo_bn(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_wo_bn.pth.tar")
model_bin_wo_bn.load_state_dict(checkpoint['state_dict'])
model_bin_wo_bn.eval()

model_real = models.__dict__["resnet"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_real = model_real(**model_config)
checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar")
model_real.load_state_dict(checkpoint['state_dict'])
model_real.eval()

layer_gc_bin_w_bn = LayerGradCam(model_bin_w_bn, list(model_bin_w_bn.children())[-6][-1].conv2)
layer_gc_bin_wo_bn = LayerGradCam(model_bin_wo_bn, list(model_bin_wo_bn.children())[-6][-1].conv2)
layer_gc_real = LayerGradCam(model_real, list(model_real.children())[-3][-1].conv2)


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

indices = []
while len(indices) < 10:
    idx = np.random.randint(0, len(testset))
    x = testset[idx][0].unsqueeze(0)
    y = testset[idx][1]
    with torch.no_grad():
        pred_bin_w_bn = model_bin_w_bn(x).argmax(dim=1).item()
        pred_bin_wo_bn = model_bin_wo_bn(x).argmax(dim=1).item()
        pred_real = model_real(x).argmax(dim=1).item()
        if pred_bin_w_bn == y and pred_bin_wo_bn == y and pred_real == y and idx not in indices:
            indices.append(idx)

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
model_bin_w_bn.to("cuda")
model_bin_wo_bn.to("cuda")
model_real.to("cuda")


plt.figure(figsize=(15, 25))

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


for idx in range(10):
    x = images[idx].unsqueeze(0).requires_grad_()
    y = labels[idx].item()

    
    
    attr_gc_bin_w_bn = layer_gc_bin_w_bn.attribute(x, target=y)
    attr_gc_bin_w_bn = F.interpolate(attr_gc_bin_w_bn, size=(32, 32), mode='bilinear', align_corners=False)
    attr_gc_bin_wo_bn = layer_gc_bin_wo_bn.attribute(x, target=y)
    attr_gc_bin_wo_bn = F.interpolate(attr_gc_bin_wo_bn, size=(32, 32), mode='bilinear', align_corners=False)
    attr_gc_real = layer_gc_real.attribute(x, target=y)
    attr_gc_real = F.interpolate(attr_gc_real, size=(32, 32), mode='bilinear', align_corners=False)

    attr_gc_bin_w_bn = clamp_attr(attr_gc_bin_w_bn)
    attr_gc_bin_wo_bn = clamp_attr(attr_gc_bin_wo_bn)
    attr_gc_real = clamp_attr(attr_gc_real)
    
    # 標題 (顯示正確與否)
    gt_name = CIFAR10_CLASSES[y]
    title = f"GT: {gt_name}"


    # 畫圖 (4 張)
    plt.subplot(10, 4, idx * 4 + 1)
    plt.imshow(tensor_to_img(vis_x[idx].cpu()))
    plt.title(title, fontsize=15)
    plt.axis("off")


    plt.subplot(10, 4, idx * 4 + 2)
    attr_gc_real = torch.tensor(attr_gc_real)
    plt.imshow(tensor_to_img(vis_x[idx].cpu()))
    plt.imshow(tensor_to_img(attr_gc_real), cmap="seismic",alpha=0.5)
    plt.title("Real-valued CNN", fontsize=15)
    plt.axis("off")
    
    plt.subplot(10, 4, idx * 4 + 3)
    attr_gc_bin_wo_bn = torch.tensor(attr_gc_bin_wo_bn)
    plt.imshow(tensor_to_img(vis_x[idx].cpu()))
    plt.imshow(tensor_to_img(attr_gc_bin_wo_bn), cmap="seismic",alpha=0.5)
    plt.title("Binary CNN without BN", fontsize=15)
    plt.axis("off")
    
    plt.subplot(10, 4, idx * 4 + 4)
    attr_gc_bin_w_bn = torch.tensor(attr_gc_bin_w_bn)
    plt.imshow(tensor_to_img(vis_x[idx].cpu()))
    plt.imshow(tensor_to_img(attr_gc_bin_w_bn), cmap="seismic",alpha=0.5)
    plt.title("Binary CNN with BN", fontsize=15)
    plt.axis("off")
    

plt.tight_layout()
plt.savefig(f"Grad_CAM_Comparison.png")
