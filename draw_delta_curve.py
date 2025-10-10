# === 2. 匯入套件 ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
import models

# === 3. Dataset (CIFAR-10) ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# === 4. Normalize / Denormalize (ImageNet stats for pretrained models) ===
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def normalize(img):
    mean = torch.tensor(__imagenet_stats['mean']).view(3,1,1)
    std = torch.tensor(__imagenet_stats['std']).view(3,1,1)
    return (img - mean) / std

def tensor_to_img(t):
    t = t.cpu().detach().numpy()
    if t.ndim == 3:
        t = np.transpose(t, (1, 2, 0))
    return np.clip(t, 0, 1)


# === 5. 載入模型 (用預訓練 ResNet18 for demo) ===
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': "cifar10"}
model = model(**model_config)
checkpoint_bin = torch.load(
    "C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar"
)
model.load_state_dict(checkpoint_bin['state_dict'])
model.eval()
model.to(device)

# === 6. 抓取 activations ===
def get_activations(model, x):
    activations = {}
    activations_inp = {}
    hooks = []

    def save_output(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
            activations_inp[name] = inp[0].detach().cpu()
        return hook

    for name, module in model.named_modules():
        # if isinstance(module, (nn.Hardtanh)):
        #     hooks.append(module.register_forward_hook(save_output(name)))
        if name == "bn3":
            hooks.append(module.register_forward_hook(save_output(name)))
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return activations, activations_inp

# === 7. baseline (挑一張測試圖) ===
img_raw, label = cifar10_test[123]
x_orig = normalize(img_raw).unsqueeze(0).to(device)

acts_orig, acts_orig_inp = get_activations(model, x_orig)

layer_names = list(acts_orig.keys())
layer_names_inp = list(acts_orig_inp.keys())

print(f"原始圖標籤: {CIFAR10_CLASSES[label]}")

import torch
import matplotlib.pyplot as plt

def pixel_response_curve_batch_relative(
    img, r, c, ch, model,
    delta_max=1.0, step=0.0001, device="cuda", batch_size=256
):
    """
    探索單一 pixel 的輸出變化曲線（Δy/Δx 相對 baseline）
    """
    # baseline 輸出
    x_orig = normalize(img).unsqueeze(0).to(device)
    with torch.no_grad():
        acts_orig, _ = get_activations(model, x_orig)
        y_orig = acts_orig["bn3"][0]
    pred_class = torch.argmax(y_orig).item()

    # 掃描範圍
    deltas = torch.arange(-delta_max, delta_max + step, step)
    num_steps = len(deltas)
    ys_list = []

    with torch.no_grad():
        for i in range(0, num_steps, batch_size):
            batch_deltas = deltas[i:i + batch_size]
            img_batch = img.unsqueeze(0).repeat(len(batch_deltas), 1, 1, 1)

            # 修改 pixel
            img_batch[torch.arange(len(batch_deltas)), ch, r, c] = torch.clamp(
                img_batch[torch.arange(len(batch_deltas)), ch, r, c] + batch_deltas, 0, 1
            )

            x_batch = normalize(img_batch).to(device)
            acts_batch, _ = get_activations(model, x_batch)
            ys_list.append(acts_batch["bn3"].cpu())

    ys = torch.cat(ys_list, dim=0)  # [num_steps, num_classes]

    # baseline index （Δx=0）
    base_idx = (deltas == 0).nonzero(as_tuple=True)[0].item()
    y_base = ys[base_idx]  # [num_classes]

    # 相對 baseline 的 Δy/Δx
    dy_dx = (ys - y_base)
    dy_dx[base_idx] = 0.0  # 避免除以 0
    
    
    y_curve = ys[:, pred_class]
    dy_dx_curve = dy_dx[:, pred_class]
    
    ys_after_softmax = F.softmax(ys, dim=1)
    y_base_after_softmax = ys_after_softmax[base_idx]
    dy_dx_after_softmax = (ys_after_softmax - y_base_after_softmax)
    dy_dx_after_softmax[base_idx] = 0.0
    
    y_curve_after_softmax = ys_after_softmax[:, pred_class]
    dy_dx_curve_after_softmax = dy_dx_after_softmax[:, pred_class]

    return deltas, y_curve.cpu(), dy_dx_curve.cpu(), y_curve_after_softmax.cpu(), dy_dx_curve_after_softmax.cpu(), pred_class


def plot_pixel_response_relative(
    img, r, c, ch, model,idx,
    delta_max=0.1, step=0.001, device="cuda", batch_size=256
):
    deltas, y_curve, dy_dx_curve, y_curve_after_softmax, dy_dx_curve_after_softmax, pred_class = pixel_response_curve_batch_relative(
        img, r, c, ch, model, delta_max, step, device, batch_size
    )

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))

    ax[0].plot(deltas.numpy(), y_curve.numpy())
    # ax[0].set_xticks(torch.linspace(-delta_max, delta_max, 25).numpy())  # 顯示更多刻度
    # ax[0].set_xlim(-delta_max, delta_max)
    ax[0].set_title(f"model output(ch={ch}, pos=({r},{c}), class={CIFAR10_CLASSES[pred_class]})")
    ax[0].set_xlabel("Pixel Change (Δx)")
    ax[0].set_ylabel("Model Output (y)")

    ax[1].plot(deltas.numpy(), dy_dx_curve.numpy())
    # ax[1].set_xticks(torch.linspace(-delta_max, delta_max, 25).numpy())  # 顯示更多刻度
    # ax[1].set_xlim(-delta_max, delta_max)
    ax[1].set_title("model output difference")
    ax[1].set_xlabel("Pixel Change (Δx)")
    ax[1].set_ylabel("Model Output Difference (Δy)")

    ax[2].plot(deltas.numpy(), y_curve_after_softmax.numpy())
    ax[2].set_title("model output after softmax")
    ax[2].set_xlabel("Pixel Change (Δx)")
    ax[2].set_ylabel("Model Output after softmax(y)")

    ax[3].plot(deltas.numpy(), dy_dx_curve_after_softmax.numpy())
    ax[3].set_title("model output difference after softmax")
    ax[3].set_xlabel("Pixel Change (Δx)")
    ax[3].set_ylabel("Model Output Difference after softmax (Δy)")

    plt.tight_layout()
    if not os.path.exists(f"curves/{idx}"):
        os.makedirs(f"curves/{idx}")
    plt.savefig(f"curves/{idx}/position({r}_{c})_ch({ch})_step{step}.png", bbox_inches="tight", dpi=150)
    # plt.show()
    plt.close()

# === Example usage ===
indices = [123]
for idx in indices:
    for ch in range(1):
        for c in range(32):
            for r in range(32):
                plot_pixel_response_relative(img_raw,r, c, ch, model,idx = idx, delta_max=1.0, step=0.001, device=device)
# r, c, ch = 16, 16, 0
# plot_pixel_response_batch(img_raw, r, c, ch, model, delta_max=1.0, step=0.0001, mode="pred", normalize=normalize, device=device)
