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
        if name == "bn3":
            hooks.append(module.register_forward_hook(save_output(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return activations, activations_inp

change_range = []
avg_pred_values = []
avg_pred_values_softmax = []
first_and_second_diff = []
num_change_pred = []

# === 7. baseline (挑一張測試圖) ===
def pixel_sensitivity_statistics(
    img, 
    delta=0.01, 
    batch_size=512,
):
   
    H, W = img.shape[1], img.shape[2]

    # baseline (無擾動)
    x_orig = normalize(img).unsqueeze(0).to(device)
    acts_orig, _ = get_activations(model, x_orig)
    y_orig = acts_orig["bn3"][0]
    pred_class = torch.argmax(y_orig).item()

    ## record baseline prediction value
    avg_pred_values.append(y_orig[pred_class].item())
    first_and_second_diff.append((y_orig[pred_class]-y_orig[torch.argsort(y_orig, descending=True)[1]]).item())
    avg_pred_values_softmax.append(F.softmax(y_orig, dim=0)[pred_class].item())
    
    
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
    changed = []
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
            
            y_mod_pred = torch.argmax(y_mod, dim=1)
            changed_batch = (y_mod_pred != pred_class)
            changed.append(changed_batch)
    changed = torch.cat(changed)
    scores = torch.cat(scores)

    change_range.append(scores.abs().mean().item())
    if changed.any():
        num_change_pred.append(1)
    


for i in range(len(cifar10_test)):
    print(i)
    img_raw, label = cifar10_test[i]
    pixel_sensitivity_statistics(img_raw, delta=0.01)
    
print("Average of prediction value:", np.mean(avg_pred_values))
print("Average of first and second difference:", np.mean(first_and_second_diff))
print("Number of images that change prediction:", len(num_change_pred))
print("Average of change range:", np.mean(change_range))
print("Standard deviation of change range:", np.std(change_range))
print("average of prediction value after softmax:", np.mean(avg_pred_values_softmax))