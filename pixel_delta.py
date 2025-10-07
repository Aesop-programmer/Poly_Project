# === 2. 匯入套件 ===
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
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(save_output(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return activations, activations_inp

# === 7. baseline (挑一張測試圖) ===
def pixel_sensitivity_map(
    img, 
    delta=0.05, 
    mode="avg", 
    batch_size=128,
    use_smoothgrad=False,
    n_samples=20,
    noise_std=0.1
):
    """
    計算 pixel-level sensitivity map (可選 SmoothGrad)
    
    Args:
        img: 單張輸入影像 (Tensor [3,H,W])
        delta: 每次 pixel 改變幅度
        mode: "avg" -> 所有 class 取絕對值平均 | "pred" -> 僅針對預測 class
        batch_size: forward 時 batch 大小
        use_smoothgrad: 是否使用 SmoothGrad
        n_samples: SmoothGrad 疊加次數
        noise_std: SmoothGrad 雜訊強度 (標準差)
    """
    H, W = img.shape[1], img.shape[2]

    # baseline (無擾動)
    x_orig = normalize(img).unsqueeze(0).to(device)
    acts_orig, _ = get_activations(model, x_orig)
    y_orig = acts_orig["fc"][0]
    pred_class = torch.argmax(y_orig).item()

    # 準備 pixel+channel perturbations 組合
    coords = [(r, c, ch) for r in range(H) for c in range(W) for ch in range(3)]

    def compute_sensitivity_for_image(img_input):
        perturbed_imgs = []
        for (r, c, ch) in coords:
            img_mod = img_input.clone()
            img_mod[ch, r, c] = min(1.0, img_mod[ch, r, c] + delta)
            perturbed_imgs.append(normalize(img_mod).unsqueeze(0))

        perturbed_imgs = torch.cat(perturbed_imgs).to(device)

        scores = []
        with torch.no_grad():
            for i in range(0, len(perturbed_imgs), batch_size):
                batch = perturbed_imgs[i:i + batch_size]
                acts_mod, _ = get_activations(model, batch)
                y_mod = acts_mod["fc"]
                dy = (y_mod - y_orig.unsqueeze(0)) / delta

                if mode == "avg":
                    score = dy.abs().mean(dim=1)
                elif mode == "pred":
                    score = dy[:, pred_class]
                else:
                    raise ValueError("mode must be 'avg' or 'pred'")
                scores.append(score.cpu())

        scores = torch.cat(scores)

        # 聚合回 HxW
        sens_map = torch.zeros((H, W))
        count = torch.zeros((H, W))
        for (r, c, ch), s in zip(coords, scores):
            sens_map[r, c] += s.item()
            count[r, c] += 1
        sens_map /= count
        return sens_map

    # === 普通版本 ===
    if not use_smoothgrad:
        sens_map = compute_sensitivity_for_image(img)
        return sens_map, pred_class

    # === SmoothGrad 版本 ===
    sensitivity_accum = torch.zeros((H, W))
    for n in range(n_samples):
        noise = torch.randn_like(img) * noise_std
        img_noisy = torch.clamp(img + noise, 0, 1)
        sens_map_noisy = compute_sensitivity_for_image(img_noisy)
        sensitivity_accum += sens_map_noisy

    sens_map_final = sensitivity_accum / n_samples
    return sens_map_final, pred_class



def normalize_map(sens_map, method="linear", gamma=2.0, alpha=10.0):
    """
    method = "linear"  -> min-max normalize
    method = "gamma"   -> power law normalization
    method = "log"     -> log normalization
    method = "sigmoid" -> z-score + sigmoid
    """
    x = sens_map.clone()
    minv, maxv = x.min(), x.max()
    if maxv == minv:
        return torch.zeros_like(x)

    if method == "linear":
        x = (x - minv) / (maxv - minv)

    elif method == "gamma":
        x = (x - minv) / (maxv - minv)
        x = x.pow(gamma)

    elif method == "log":
        x = x - minv
        x = torch.log1p(alpha * x) / torch.log1p(alpha * (maxv - minv))

    elif method == "sigmoid":
        mu, sigma = x.mean(), x.std()
        x = torch.sigmoid((x - mu) / (sigma + 1e-8))

    return x

def keep_high_low(sens_map, thr=95):
    threshold = np.percentile(np.abs(sens_map),thr)
    mask = (sens_map >=threshold) | (sens_map <= -threshold)
    sens_map = sens_map*mask
    return sens_map


def visualize_sensitivity(img, sens_map, pred_class=None, mode="avg",thr=None,delta=0.05,idx=0,smooth=False,std=None):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    sens_map_disp = sens_map.clone()

    # optional normalize
    if not(thr is None):
        sens_map_disp = keep_high_low(sens_map_disp, thr=thr)
    # threshold = np.percentile(np.abs(sens_map_disp),thr)
    # sens_map_disp = np.clip(sens_map_disp, -threshold, threshold)
        
    
    ax[0].imshow(tensor_to_img(img))
    ax[0].set_title("Original Image_pred_class="+str(CIFAR10_CLASSES[pred_class]), fontsize=15)
    ax[0].axis("off")
    
    
    # Heatmap
    im1 = ax[1].imshow(sens_map_disp.abs().numpy(), cmap="gray")
    ax[1].set_title(f" mode={mode}, delta={delta}")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].axis("off")

    # Overlay on original
    img_disp = tensor_to_img(img)
    ax[2].imshow(img_disp)
    im2 = ax[2].imshow(sens_map_disp.numpy(), cmap="seismic", alpha=0.65)
    ax[2].set_title(f"(Pred={CIFAR10_CLASSES[pred_class]}), mode={mode}, delta={delta}")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].axis("off")
    if smooth:
        if thr is None:
            plt.savefig(f"BNN_delta/idx_{idx}_delta_mode_{mode}_delta_{delta}_smooth_{std}.png", bbox_inches="tight", dpi=150)
        else:
            plt.savefig(f"BNN_delta/idx_{idx}_delta_mode_{mode}_delta_{delta}_thr_{thr}_smooth_{std}.png", bbox_inches="tight", dpi=150)
    else:  
        if thr is None:
            plt.savefig(f"BNN_delta/idx_{idx}_delta_mode_{mode}_delta_{delta}.png", bbox_inches="tight", dpi=150)
        else:
            plt.savefig(f"BNN_delta/idx_{idx}_delta_mode_{mode}_delta_{delta}_thr_{thr}.png", bbox_inches="tight", dpi=150)
    # plt.show()
    
# import random
# # 隨機挑20 
# random_indices = random.sample(range(len(cifar10_test)), 20)
# for idx in random_indices:
#     img_raw, label = cifar10_test[idx]
#     x_orig = normalize(img_raw).unsqueeze(0).to(device)

#     acts_orig, acts_orig_inp = get_activations(model, x_orig)
#     y_orig = acts_orig["fc"][0]
#     pred_class = torch.argmax(y_orig).item()
    
#     sens_map_avg, pred_class = pixel_sensitivity_map(img_raw, delta=0.015, mode="avg")
#     # sens_map_avg = normalize_map(sens_map_avg, method="linear")
#     # sens_map_avg = keep_high_low(sens_map_avg, thr= 50)
#     visualize_sensitivity(img_raw, sens_map_avg, pred_class, mode="avg",delta=0.015,idx=idx)
#     visualize_sensitivity(img_raw, sens_map_avg, pred_class, mode="avg",thr=50,delta=0.015,idx=idx)

delta = [0.01, 0.015, 0.02, 0.03, 0.05,]
std = [0.05, 0.1, 0.15]
img_raw, label = cifar10_test[123]
x_orig = normalize(img_raw).unsqueeze(0).to(device)

acts_orig, acts_orig_inp = get_activations(model, x_orig)
y_orig = acts_orig["fc"][0]
pred_class = torch.argmax(y_orig).item()
for d in delta:
    for s in std:
        sens_map_avg, pred_class = pixel_sensitivity_map(img_raw, delta=d, mode="avg",use_smoothgrad=True,n_samples=50,noise_std=s)
        # sens_map_avg = normalize_map(sens_map_avg, method="linear")
        # sens_map_avg = keep_high_low(sens_map_avg, thr= 50)
        visualize_sensitivity(img_raw, sens_map_avg, pred_class, mode="avg",delta=d,idx=123,smooth=True,std=s)

        sens_map_avg, pred_class = pixel_sensitivity_map(img_raw, delta=d, mode="pred",use_smoothgrad=True,n_samples=50,noise_std=s)
        visualize_sensitivity(img_raw, sens_map_avg, pred_class, mode="pred",delta=d,idx=123,smooth=True,std=s)
        # visualize_sensitivity(img_raw, sens_map_avg, pred_class, mode="avg",thr=50,delta=d,idx=123)