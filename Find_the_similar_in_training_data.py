import torch, numpy as np, matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms


# === 2. 匯入套件 ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import models

# === 3. Dataset (CIFAR-10) ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
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
    "/users/tseng/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar"
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
        if name == "tanh2":
            hooks.append(module.register_forward_hook(save_output(name)))
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()
    return activations, activations_inp

def find_similar_images(test_idx, train_dataset, model, top_k=5, batch_size=256):
    test_img, test_label = cifar10_test[test_idx]
    test_img = normalize(test_img.unsqueeze(0)).to(device)
    test_activations, _ = get_activations(model, test_img)
    test_act = test_activations["tanh2"].squeeze().numpy()

    similarities = []
    with torch.no_grad():
        for i in range(0, len(train_dataset), batch_size):
            train_imgs = [normalize(train_dataset[j][0].unsqueeze(0)).to(device) for j in range(i, min(i + batch_size, len(train_dataset)))]
            train_activations, _ = get_activations(model, torch.cat(train_imgs))
            train_act = train_activations["tanh2"].squeeze().numpy()
            
            # 因為他們都是0 和 1 的激活值，所以比較的基準是他看他們有幾根不一樣
            sim = np.sum(np.sign(test_act) != np.sign(train_act), axis=1)
            for s, idx in zip(sim, range(i, min(i + batch_size, len(train_dataset)))):
                similarities.append((s, idx))
    

    similarities.sort(key=lambda x: x[0])
    top_k_indices = [(s, idx) for s, idx in similarities[:top_k]]
    
    # # 畫 test_act 的output
    # figure, ax = plt.subplots()
    # ax.bar(range(len(test_act)), np.sign(test_act), label="Test Image Activations", alpha=0.7)
    # ax.set_title("Distribution of Test Image Activations")
    # ax.set_xlabel("Channels")
    # ax.set_ylabel("Activation Value")
    # ax.legend()
    # plt.savefig(f"{PATH}/test_idx_{test_idx}_activations.png")
    # plt.close()
    # exit()    
    
    fig, ax = plt.subplots(1, top_k + 1, figsize=(15, 3))
    ax[0].imshow(tensor_to_img(test_img.squeeze().cpu()))
    ax[0].set_title(f"Test Image label: {CIFAR10_CLASSES[test_label]}\n, Avg diff: {np.mean([s for s, _ in similarities]):.4f}\n pred: {CIFAR10_CLASSES[model(test_img).argmax(dim=1).item()]}")
    ax[0].axis('off')
    for i, (s, idx) in enumerate(top_k_indices):
        img, label = train_dataset[idx]
        ax[i + 1].imshow(tensor_to_img(img))
        ax[i + 1].set_title(f"Idx: {idx}\nLabel: {CIFAR10_CLASSES[label]}\nDiff: {s:.4f}\npred: {CIFAR10_CLASSES[model(normalize(img.unsqueeze(0)).to(device)).argmax(dim=1).item()]}")
        ax[i + 1].axis('off')

        # # 每個img 下面寫出他們是差在哪些index
        # train_activations, _ = get_activations(model, normalize(img.unsqueeze(0)).to(device))
        # train_act = train_activations["tanh2"].squeeze().numpy()
        # diff_indices = np.where(np.sign(test_act) != np.sign(train_act))[0]
        # ax[1][i+1].text(0.1, 0.5, f"Differing Indices:\n{[if x for x in diff_indices.tolist()]}", fontsize=10)
        # ax[1][i+1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{PATH}/test_idx_{test_idx}.png")
    plt.close()
    return top_k_indices

import random
#generate 30 examples
PATH = "./similarity_results/"
import os
os.makedirs(PATH, exist_ok=True)
test_indx = random.sample(range(0, len(cifar10_test)-1), 30)
for test_idx in test_indx:
    find_similar_images(test_idx, cifar10_train, model, top_k=5)

