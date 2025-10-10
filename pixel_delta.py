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
        if isinstance(module, (nn.Hardtanh)):
            hooks.append(module.register_forward_hook(save_output(name)))
        if name == "bn3":
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
    batch_size=128,
):
   
    H, W = img.shape[1], img.shape[2]

    # baseline (無擾動)
    x_orig = normalize(img).unsqueeze(0).to(device)
    acts_orig, _ = get_activations(model, x_orig)
    y_orig = acts_orig["bn3"][0]
    pred_class = torch.argmax(y_orig).item()

    # 準備 pixel+channel perturbations 組合
    coords = [(r, c, ch) for r in range(H) for c in range(W) for ch in range(3)]

    perturbed_imgs = []
    for (r, c, ch) in coords:
        img_mod = img.clone()
        img_mod[ch, r, c] = min(1.0, img_mod[ch, r, c] + delta)
        perturbed_imgs.append(normalize(img_mod).unsqueeze(0))

    perturbed_imgs = torch.cat(perturbed_imgs).to(device)

    scores = []
    scores_softmax = []
    with torch.no_grad():
        for i in range(0, len(perturbed_imgs), batch_size):
            batch = perturbed_imgs[i:i + batch_size]
            acts_mod, _ = get_activations(model, batch)
            y_mod = acts_mod["bn3"]
            dy = (y_mod - y_orig.unsqueeze(0))
            score = dy[:, pred_class]
            scores.append(score)
            dy_softmax = F.softmax(y_mod, dim=1) - F.softmax(y_orig.unsqueeze(0), dim=1)
            score_softmax = dy_softmax[:, pred_class]
            scores_softmax.append(score_softmax)
    scores = torch.cat(scores)
    scores_softmax = torch.cat(scores_softmax)
    # 聚合回 HxW
    sens_map = torch.zeros((H, W))
    count = torch.zeros((H, W))
    sens_map_softmax = torch.zeros((H, W))
    for (r, c, ch), s, s_softmax in zip(coords, scores, scores_softmax):
        sens_map[r, c] += s.item()
        sens_map_softmax[r, c] += s_softmax.item()
        count[r, c] += 1
    sens_map /= count
    sens_map_softmax /= count
    return sens_map, sens_map_softmax, pred_class

    
def visualize_sensitivity(img,delta=0.05,idx=0):
    fig, ax = plt.subplots(1, 5, figsize=(25,5))

    
    sens_map, sens_map_softmax, pred_class = pixel_sensitivity_map(img, delta=delta)
        
    
    ax[0].imshow(tensor_to_img(img))
    ax[0].set_title("Original Image_pred_class="+str(CIFAR10_CLASSES[pred_class]), fontsize=15)
    ax[0].axis("off")
    
    # Heatmap
    im1 = ax[1].imshow(sens_map.abs().numpy(), cmap="gray")
    ax[1].set_title(f"model_pred_difference_abs_heatmap, delta={delta}")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].axis("off")

    # Overlay on original
    img_disp = tensor_to_img(img)
    ax[2].imshow(img_disp)
    im2 = ax[2].imshow(sens_map.numpy(), cmap="seismic", alpha=0.65)
    ax[2].set_title(f"model_pred_difference, delta={delta}")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].axis("off")
    
    im3 = ax[3].imshow(sens_map_softmax.abs().numpy(), cmap="gray")
    ax[3].set_title(f"model_softmax_difference_abs_heatmap, delta={delta}")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].axis("off")
    
    # Overlay on original
    ax[4].imshow(img_disp)
    im4 = ax[4].imshow(sens_map_softmax.numpy(), cmap="seismic", alpha=0.65)
    ax[4].set_title(f"model_softmax_difference, delta={delta}")
    fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"BNN_delta/idx_{idx}_delta_{delta}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

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

delta = [0.00001,0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
img_raw, label = cifar10_test[123]
x_orig = normalize(img_raw).unsqueeze(0).to(device)

acts_orig, acts_orig_inp = get_activations(model, x_orig)
y_orig = acts_orig["bn3"][0]
pred_class = torch.argmax(y_orig).item()
for d in delta:
    visualize_sensitivity(img_raw, delta=d,idx=123)