import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from torchvision import datasets, transforms
import models
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Superpixel-based Sensitivity Analysis')
parser.add_argument('--analyze_mode', default='joint', type=str, help='analysis mode', choices=['individual','joint'])
parser.add_argument('--perturb_mode', type=str, help='perturbation mode', choices=['delta', 'black', 'white', 'gaussian', 'mosaic'])
parser.add_argument('--delta',type=float, default=0.01, help="delta value")
parser.add_argument('--batch_size', type=int, help='batch size for processing', default=256)
parser.add_argument('--n_segments', type=int, default=10, help='number of superpixels')
parser.add_argument('--compactness',type=int,default=10,help="compactness parameter for SLIC")
parser.add_argument('--mosaic_block',type=int,default=6)
args = parser.parse_args()


# ---------------- Config ----------------
SAVE_DIR = "./results_superpixel"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "cifar10"
ANALYZE_MODE = args.analyze_mode  # "individual" or "joint"
PERTURB_MODE = args.perturb_mode       # "delta", "black", "white", "gaussian", "mosaic"
DELTA = args.delta
BATCH_SIZE = args.batch_size
N_SEGMENTS = args.n_segments
COMPACTNESS = args.compactness
MOSAIC_BLOCK = args.mosaic_block

# ---------------- Normalization utils ----------------
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

# ---------------- Load dataset ----------------
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ---------------- Load model ----------------
model = models.__dict__["resnet_binary"]
model_config = {'input_size': 32, 'dataset': DATASET}
model = model(**model_config)
checkpoint_bin = torch.load(
    "/users/tseng/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar",
    map_location=DEVICE
)
model.load_state_dict(checkpoint_bin['state_dict'])
model.eval()
model.to(DEVICE)

# ---------------- Activation hook ----------------
def get_bn3_output(model, x):
    activations = {}
    hooks = []

    def save_output(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    for name, module in model.named_modules():
        if name == "bn3":
            hooks.append(module.register_forward_hook(save_output(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return activations["bn3"]

# ---------------- Perturbation ----------------
def apply_superpixel_modification(img_t, segments, labels_to_modify, mode="delta", delta=0.05, mosaic_block=6):
    img_mod = img_t.clone()
    H, W = segments.shape
    for lbl in labels_to_modify:
        mask = torch.tensor(segments == lbl)
        for c in range(3):
            if mode == "delta":
                img_mod[c][mask] = torch.clamp(img_mod[c][mask] + delta, 0, 1)
            elif mode == "black":
                img_mod[c][mask] = 0.0
            elif mode == "white":
                img_mod[c][mask] = 1.0
            elif mode == "gaussian":
                mean = __imagenet_stats['mean'][c]
                std = __imagenet_stats['std'][c]
                noise = torch.randn_like(img_mod[c][mask]) * std + mean
                img_mod[c][mask] = torch.clamp(noise, 0, 1)
            elif mode == "mosaic":
                ys, xs = torch.where(mask)
                if len(ys) == 0:
                    continue
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()
                region = img_mod[c, y_min:y_max+1, x_min:x_max+1].clone()
                region_mask = mask[y_min:y_max+1, x_min:x_max+1]
                h, w = region.shape
                for yy in range(0, h, mosaic_block):
                    for xx in range(0, w, mosaic_block):
                        y_end = min(yy + mosaic_block, h)
                        x_end = min(xx + mosaic_block, w)
                        block = region[yy:y_end, xx:x_end]
                        block_mask = region_mask[yy:y_end, xx:x_end]
                        if block_mask.sum() == 0:
                            continue
                        mean_val = block[block_mask].mean()
                        region[yy:y_end, xx:x_end][block_mask] = mean_val
                img_mod[c, y_min:y_max+1, x_min:x_max+1] = region
    return img_mod

# ---------------- Sensitivity computation ----------------
def compute_segment_sensitivity(img_t, segments,
                                mode="delta", delta=0.05, mosaic_block=6,
                                batch_size=64, analyze_mode="individual"):
    H, W = segments.shape
    x_orig = normalize(img_t).unsqueeze(0).to(DEVICE)
    y_orig = get_bn3_output(model, x_orig)[0].cpu()
    pred_class = int(torch.argmax(y_orig).item())

    unique_segs = np.unique(segments)
    if analyze_mode == "joint":
        # iterative each segments
        segments_to_test = unique_segs.tolist()
        img_mod_list = []
        for lbl in segments_to_test:
            img_mod = apply_superpixel_modification(img_t, segments, [lbl],
                                                    mode=mode, delta=delta, mosaic_block=mosaic_block)
            img_mod_list.append(normalize(img_mod).unsqueeze(0))
            
        imgs_mod = torch.cat(img_mod_list).to(DEVICE)
        scores, scores_soft = [], []
        with torch.no_grad():
            for i in range(0, len(imgs_mod), batch_size):
                batch = imgs_mod[i:i+batch_size]
                y_mod = get_bn3_output(model, batch).cpu()
                dy = y_mod - y_orig.unsqueeze(0)
                scores.append(dy[:, pred_class])
                soft_diff = F.softmax(y_mod, dim=1) - F.softmax(y_orig.unsqueeze(0), dim=1)
                scores_soft.append(soft_diff[:, pred_class])
        scores = torch.cat(scores).numpy()
        scores_soft = torch.cat(scores_soft).numpy()
        
        sens_map = np.zeros((H,W))
        sens_map_soft = np.zeros((H,W))
        for lbl, s, ss in zip(segments_to_test, scores, scores_soft):
            sens_map[segments==lbl] = s
            sens_map_soft[segments==lbl] = ss
        return sens_map, sens_map_soft, pred_class

    

# ---------------- Visualization ----------------
def save_visualizations(img_t, sens_map, sens_map_soft, pred_class, delta, save_path, colored):
    img_disp = tensor_to_img(img_t)
    fig, ax = plt.subplots(1,6, figsize=(30,5))
    ax[0].imshow(img_disp)
    ax[0].set_title(f"Original ({CIFAR10_CLASSES[pred_class]})")
    ax[0].axis("off")

    ax[1].imshow(colored)
    ax[1].set_title("Segmented")
    ax[1].axis("off")


    im1 = ax[2].imshow(np.abs(sens_map), cmap="gray")
    ax[2].set_title(f"pre-softmax |Δ|, mode={PERTURB_MODE}")
    fig.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].axis("off")

    ax[3].imshow(img_disp)
    im2 = ax[3].imshow(sens_map, cmap="seismic", alpha=0.65)
    ax[3].set_title("pre-softmax Δ")
    fig.colorbar(im2, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].axis("off")

    im3 = ax[4].imshow(np.abs(sens_map_soft), cmap="gray")
    ax[4].set_title(f"|softmax Δ|")
    fig.colorbar(im3, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].axis("off")

    ax[5].imshow(img_disp)
    im4 = ax[5].imshow(sens_map_soft, cmap="seismic", alpha=0.65)
    ax[5].set_title("softmax Δ")
    fig.colorbar(im4, ax=ax[5], fraction=0.046, pad=0.04)
    ax[5].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------- Main loop ----------------
def process_dataset(index):

    for i in index:
        img_t, label = dataset[i]
        img_np = np.transpose(img_t.numpy(), (1,2,0))
        segments = slic(img_np, n_segments=N_SEGMENTS, compactness=COMPACTNESS, start_label=0)
        sens_map, sens_map_soft, pred_class = compute_segment_sensitivity(
            img_t, segments, mode=PERTURB_MODE, delta=DELTA,
            mosaic_block=MOSAIC_BLOCK, batch_size=BATCH_SIZE, analyze_mode=ANALYZE_MODE
        )
        colored = label2rgb(segments, img_np, kind='avg', bg_label=None)
        save_path = os.path.join(f"{SAVE_DIR}/{PERTURB_MODE}/{N_SEGMENTS}", f"img_{i:04d}_{CIFAR10_CLASSES[pred_class]}_{PERTURB_MODE}.png")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_visualizations(img_t, sens_map, sens_map_soft, pred_class, DELTA, save_path, colored)
        
        # np.save(os.path.join(SAVE_DIR, f"img_{i:04d}_sensmap.npy"), sens_map)
        # np.save(os.path.join(SAVE_DIR, f"img_{i:04d}_sensmap_soft.npy"), sens_map_soft)

perturb_modes = ['delta', 'black', 'white', 'gaussian', 'mosaic']
n_segments_list = [10, 20, 30, 50, 100 ,300]
Delta = 0.0001
import random
random.seed(91123)
if __name__ == "__main__":
    # random generate index for 30 images
    random_indices = random.sample(range(len(dataset)), 30)
    for perturb_mode in perturb_modes:
        PERTURB_MODE = perturb_mode
        for n_segments in n_segments_list:
            N_SEGMENTS = n_segments
            process_dataset(index=random_indices)
