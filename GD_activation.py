import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import models

# === 0) Normalize 設定 ===
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**__imagenet_stats)
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# === 1) 載入 Binary ResNet ===
model_bin = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model_bin = model_bin(**model_config)
checkpoint_bin = torch.load(
    "C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar"
)
model_bin.load_state_dict(checkpoint_bin['state_dict'])
model_bin.eval()

# === 2) 載入 Real ResNet ===
model_real = models.__dict__["resnet"]
model_real = model_real(**model_config)
checkpoint_real = torch.load(
    "C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar"
)
model_real.load_state_dict(checkpoint_real['state_dict'])
model_real.eval()


# === 3) Hook 註冊 (activation + retain_grad) ===
def get_activation_dict(model, layer_types=(nn.Conv2d,)):
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            output.retain_grad()       # 保留 output 的梯度
            activations[name] = output
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, layer_types):
            layer.register_forward_hook(hook_fn(name))
    return activations


acts_bin = get_activation_dict(model_bin)
acts_real = get_activation_dict(model_real)

highest_score = None
image_best = None
label_best = None
for i in range(len(testset)):
    image, label = testset[i]
    image = image.unsqueeze(0)  # [1, 3, 32, 32]
    output_bin = model_bin(image)
    if model_real(image).argmax(1).item() == label and output_bin.argmax(1).item() == label:
        if highest_score is None or output_bin.max().item() > highest_score:
            highest_score = output_bin.max().item()
            image_best = image
            label_best = label
image = image_best
label = label_best
# === 4) pick best ===
# import random
# while True:
#     idex = random.randint(0, len(testset)-1)
#     image, label = testset[idex]
#     image = image.unsqueeze(0)  # [1, 3, 32, 32]
#     if model_real(image).argmax(1).item() == label and model_bin(image).argmax(1).item() == label:
#         break

def denormalize(img_tensor):
    mean = torch.tensor(__imagenet_stats["mean"]).view(3,1,1)
    std  = torch.tensor(__imagenet_stats["std"]).view(3,1,1)
    return img_tensor * std + mean

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

os.makedirs("results_grad_compare_w_bn", exist_ok=True)
orig = denormalize(image[0]).permute(1,2,0).numpy()
plt.imshow(orig)
plt.axis("off")
plt.title(f"Input Image, label: {CIFAR10_CLASSES[label]}")
plt.savefig("results_grad_compare_w_bn/input_image.png", bbox_inches="tight")
plt.close()
print("Saved: results_grad_compare_w_bn/input_image.png")


# === 5) Forward + Backward ===
criterion = nn.CrossEntropyLoss()

model_bin.eval()
model_real.eval()

with torch.set_grad_enabled(True):   # 確保 autograd 啟用
    out_bin = model_bin(image)
    out_real = model_real(image)

    loss_bin = criterion(out_bin, torch.tensor([label]))
    loss_real = criterion(out_real, torch.tensor([label]))

model_bin.zero_grad()
model_real.zero_grad()

loss_bin.backward()
loss_real.backward()



# === 6) 壓縮梯度方法 ===
def compress_feature(tensor, mode="maxabs"):
    """
    tensor: [1, C, H, W] (gradient)
    return: [H, W]
    """
    tensor = tensor.squeeze(0)  # [C, H, W]

    if mode == "maxabs":
        abs_tensor = torch.abs(tensor)
        idx = torch.argmax(abs_tensor, dim=0)
        out = tensor.gather(0, idx.unsqueeze(0)).squeeze(0)
    elif mode == "mean":
        out = tensor.mean(dim=0)
    elif mode == "sum":
        out = tensor.sum(dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out.numpy()


# === 7) 畫對照圖 (real vs binary 梯度) ===
def plot_and_save_compare(real_tensor, bin_tensor, layer_name, mode="maxabs", save_dir="results_grad_compare"):
    os.makedirs(save_dir, exist_ok=True)

    real_comp = compress_feature(real_tensor, mode=mode)
    bin_comp  = compress_feature(bin_tensor, mode=mode)

    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Layer {layer_name} | Grad | Mode: {mode}", fontsize=14)

    im0 = axes[0].imshow(real_comp, cmap="seismic",)
    axes[0].axis("off")
    axes[0].set_title("Real")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(bin_comp, cmap="seismic", )
    axes[1].axis("off")
    axes[1].set_title("Binary")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    save_path = os.path.join(save_dir, f"{layer_name}_{mode}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


# === 8) 執行比較 ===
modes = ["maxabs", "mean", "sum"]

for name in acts_bin.keys():
    grad_bin = acts_bin[name].grad
    grad_real = acts_real[name].grad
    if grad_bin is None or grad_real is None:
        print(f"[WARN] No gradient for {name}")
        continue

    print(f"Compare gradients for layer {name} | real: {grad_real.shape}, bin: {grad_bin.shape}")
    for mode in modes:
        plot_and_save_compare(grad_real, grad_bin, name, mode=mode, save_dir="results_grad_compare_w_bn")


# === 9) 存每個 channel 的梯度 (real + binary) ===
def save_all_gradients(tensor, layer_name, model_type, save_root="results_grad_compare"):
    tensor = tensor.squeeze(0)  # [C, H, W]
    save_dir = os.path.join(save_root, layer_name, model_type)
    os.makedirs(save_dir, exist_ok=True)

    for ch in range(tensor.shape[0]):
        grad_map = tensor[ch].cpu().numpy()
        plt.imshow(grad_map, cmap="seismic")
        plt.axis("off")
        if layer_name == "layer3.1.conv2" and model_type == "binary":
            plt.title(f"{layer_name} | {model_type} | grad ch={ch} | value={grad_map.min()}")
            print(grad_map)
        else:
            plt.title(f"{layer_name} | {model_type} | grad ch={ch}")
        
        plt.colorbar(fraction=0.046, pad=0.04)
        save_path = os.path.join(save_dir, f"grad_ch_{ch}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    print(f"Saved {tensor.shape[0]} gradient maps at: {save_dir}")


# === 10) 執行 channel-wise 存檔 ===
for name in acts_bin.keys():
    grad_bin = acts_bin[name].grad
    grad_real = acts_real[name].grad
    if grad_bin is None or grad_real is None:
        continue

    save_all_gradients(grad_real, name, "real", save_root="results_grad_compare_w_bn")
    save_all_gradients(grad_bin, name, "binary", save_root="results_grad_compare_w_bn")

## GradCAM result
import torch.nn.functional as F
from captum.attr import LayerGradCam

layer_gc_real = LayerGradCam(model_real, list(model_real.children())[-3][-1].conv2)
layer_gc_bn = LayerGradCam(model_bin, list(model_bin.children())[-6][-1].conv2)

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
    threshold = np.percentile(np.abs(attr), 95)
    attr = np.clip(attr, -threshold, threshold)
    
    return attr



attr_gc_real = layer_gc_real.attribute(image, target=label)
attr_gc_real = F.interpolate(attr_gc_real, size=(32, 32), mode='bilinear', align_corners=False)

attr_gc_bin = layer_gc_bn.attribute(image, target=label)
attr_gc_bin = F.interpolate(attr_gc_bin, size=(32, 32), mode='bilinear', align_corners=False)

attr_gc_real = clamp_attr(attr_gc_real)
attr_gc_bin = clamp_attr(attr_gc_bin)

attr_gc_real = torch.tensor(attr_gc_real)
plt.imshow(tensor_to_img(denormalize(image).cpu()))
plt.imshow(tensor_to_img(attr_gc_real), cmap="seismic",alpha=0.5)
plt.title("GradCAM_real", fontsize=15)
plt.axis("off")
plt.savefig("results_grad_compare_w_bn/GradCAM_real.png", bbox_inches="tight", dpi=150)
plt.close()


attr_gc_bin = torch.tensor(attr_gc_bin)
plt.imshow(tensor_to_img(denormalize(image).cpu()))
plt.imshow(tensor_to_img(attr_gc_bin), cmap="seismic",alpha=0.5)
plt.title("GradCAM_bin", fontsize=15)
plt.axis("off")
plt.savefig("results_grad_compare_w_bn/GradCAM_bin.png", bbox_inches="tight", dpi=150)
plt.close()