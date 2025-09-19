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
    "/home/aesop/BNN/Gradient_based/BinaryNet.pytorch/results/2025-08-30_03-10-40/model_best.pth.tar"
)
model_bin.load_state_dict(checkpoint_bin['state_dict'])
model_bin.eval()

# === 2) 載入 Real ResNet ===
model_real = models.__dict__["resnet"]
model_real = model_real(**model_config)
checkpoint_real = torch.load(
    "/home/aesop/BNN/Gradient_based/BinaryNet.pytorch/results/2025-08-30_03-13-26/model_best.pth.tar"
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


# === 4) 測試圖片 ===
import random
idex = random.randint(0, len(testset)-1)
image, label = testset[idex]
image = image.unsqueeze(0)  # [1, 3, 32, 32]

def denormalize(img_tensor):
    mean = torch.tensor(__imagenet_stats["mean"]).view(3,1,1)
    std  = torch.tensor(__imagenet_stats["std"]).view(3,1,1)
    return img_tensor * std + mean

os.makedirs("results_grad_compare", exist_ok=True)
orig = denormalize(image[0]).permute(1,2,0).numpy()
plt.imshow(orig)
plt.axis("off")
plt.title("Input Image")
plt.savefig("results_grad_compare/input_image.png", bbox_inches="tight")
plt.close()
print("Saved: results_grad_compare/input_image.png")


# === 5) Forward + Backward ===
criterion = nn.CrossEntropyLoss()

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

    im0 = axes[0].imshow(real_comp, cmap="seismic", vmin=vmin, vmax=vmax)
    axes[0].axis("off")
    axes[0].set_title("Real")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(bin_comp, cmap="seismic", vmin=vmin, vmax=vmax)
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
        plot_and_save_compare(grad_real, grad_bin, name, mode=mode, save_dir="results_grad_compare")


# === 9) 存每個 channel 的梯度 (real + binary) ===
def save_all_gradients(tensor, layer_name, model_type, save_root="results_grad_compare"):
    tensor = tensor.squeeze(0)  # [C, H, W]
    save_dir = os.path.join(save_root, layer_name, model_type)
    os.makedirs(save_dir, exist_ok=True)

    for ch in range(tensor.shape[0]):
        grad_map = tensor[ch].cpu().numpy()
        plt.imshow(grad_map, cmap="seismic")
        plt.axis("off")
        plt.title(f"{layer_name} | {model_type} | grad ch={ch}")
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

    save_all_gradients(grad_real, name, "real", save_root="results_grad_compare")
    save_all_gradients(grad_bin, name, "binary", save_root="results_grad_compare")
