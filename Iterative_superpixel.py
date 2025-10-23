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
parser.add_argument('--mosaic_block',type=int,default=2)
args = parser.parse_args()


# ---------------- Config ----------------
SAVE_DIR = "./results_iterative_superpixel_v2_no_earlystopscore"
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
def apply_superpixel_modification(img_t, segments, labels_to_modify, mode="delta", delta=0.05, mosaic_block=2):
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


    

# ---------------- Visualization ----------------
def save_visualizations(img_t, sens_map_soft, pred_class, delta, save_path, segments, best_R):
    img_disp = tensor_to_img(img_t)
    fig, ax = plt.subplots(1,5, figsize=(15,5))
    ax[0].imshow(img_disp)
    ax[0].set_title(f"Original ({CIFAR10_CLASSES[pred_class]})")
    ax[0].axis("off")

    colored = label2rgb(segments, img_disp, kind='avg', bg_label=None)
    ax[1].imshow(colored)
    ax[1].set_title(f"Superpixel Segmentation (n={N_SEGMENTS})")
    ax[1].axis("off")

    modified_img = apply_superpixel_modification(img_t, segments, best_R.labels,
                                                mode=PERTURB_MODE, delta=delta, mosaic_block=MOSAIC_BLOCK)
    modified_img_disp = tensor_to_img(modified_img)
    ax[2].imshow(modified_img_disp)
    ax[2].set_title(f"Modified Image ({PERTURB_MODE}")
    ax[2].axis("off")

    im3 = ax[3].imshow(np.abs(sens_map_soft), cmap="gray")
    ax[3].set_title(f"|softmax Δ|")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].axis("off")

    ax[4].imshow(img_disp)
    im4 = ax[4].imshow(sens_map_soft, cmap="seismic", alpha=0.65)
    ax[4].set_title("softmax Δ")
    fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------- Main loop ----------------
def process_dataset(index):

    for i in index:
        img_t, label = dataset[i]
        img_np = np.transpose(img_t.numpy(), (1,2,0))
        segments = slic(img_np, n_segments=N_SEGMENTS, compactness=COMPACTNESS, start_label=0)
        sens_map_soft, pred_class, best_R = adaptive_region_growth(
            img_t, segments,model=model, mode=PERTURB_MODE, delta=DELTA,
            mosaic_block=MOSAIC_BLOCK, batch_size=BATCH_SIZE, max_iter=MAX_iter
        )
        save_path = os.path.join(f"{SAVE_DIR}/{PERTURB_MODE}/{N_SEGMENTS}", f"img_{i:04d}_{CIFAR10_CLASSES[pred_class]}_{PERTURB_MODE}_{MAX_iter}.png")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_visualizations(img_t, sens_map_soft, pred_class, DELTA, save_path, segments, best_R)
        
        # np.save(os.path.join(SAVE_DIR, f"img_{i:04d}_sensmap.npy"), sens_map)
        # np.save(os.path.join(SAVE_DIR, f"img_{i:04d}_sensmap_soft.npy"), sens_map_soft)




import random
from collections import defaultdict

# ---------------- 新增: 區域物件 ----------------
class Region:
    def __init__(self, labels):
        self.labels = set(labels)
        self.score = None  # 模型輸出改變程度
        self.neighbors = set()
        self.num_pixels = 0
        
    def update_neighbors(self, adjacency):
        self.neighbors = set()
        for lbl in self.labels:
            self.neighbors |= adjacency[lbl]
        self.neighbors -= self.labels  # 不包含自己


# ---------------- 新增: 建立 superpixel adjacency ----------------
def build_adjacency(segments):    
    H, W = segments.shape
    adjacency = defaultdict(set)
    for y in range(H - 1):
        for x in range(W - 1):
            a, b, c = segments[y, x], segments[y+1, x], segments[y, x+1]
            adjacency[a].update([b, c])
            adjacency[b].update([a, c])
            adjacency[c].update([a, b])
    return adjacency


# ---------------- 新增: 批次評估 ----------------
def evaluate_regions(img_t, segments, regions, model, baseline_y, pred_class,
                     mode="delta", delta=0.05, mosaic_block=2, batch_size=64):
    imgs_mod = []
    for R in regions:
        img_mod = apply_superpixel_modification(img_t, segments, R.labels,
                                                mode=mode, delta=delta, mosaic_block=mosaic_block)
        imgs_mod.append(normalize(img_mod).unsqueeze(0))

    imgs_mod = torch.cat(imgs_mod).to(DEVICE)
    scores = []
    with torch.no_grad():
        for i in range(0, len(imgs_mod), batch_size):
            batch = imgs_mod[i:i+batch_size]
            y_mod = get_bn3_output(model, batch).cpu()
            # dy = y_mod - baseline_y.unsqueeze(0)
            dy = F.softmax(y_mod, dim=1) - F.softmax(baseline_y.unsqueeze(0), dim=1)
            scores.append(dy[:, pred_class])
    scores = torch.cat(scores).numpy()
    for R, s in zip(regions, scores):
        R.score = float(s)
    return regions

def connect_graph(labels_set, adjacency):
    # 檢查 labels_set 是否連通
    visited = set()
    to_visit = {next(iter(labels_set))}
    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        for neighbor in adjacency[current]:
            if neighbor in labels_set and neighbor not in visited:
                to_visit.add(neighbor)
    return visited == labels_set


# ---------------- 新增: 主疊代演算法 ----------------
def adaptive_region_growth(img_t, segments, model,
                            max_iter=200,
                           delta=0.05, mosaic_block=2,
                           mode="delta", batch_size=256):
    H, W = segments.shape
    unique_labels = np.unique(segments)
    adjacency = build_adjacency(segments)

    # baseline
    x_orig = normalize(img_t).unsqueeze(0).to(DEVICE)
    baseline_y = get_bn3_output(model, x_orig)[0].cpu()
    pred_class = int(torch.argmax(baseline_y).item())

    # Initialized all segments as regions
    regions = [Region([lbl]) for lbl in unique_labels]
    # Initialized the number of pixel in each region
    for R in regions:
        R.num_pixels = np.sum(segments == list(R.labels)[0])
        R.update_neighbors(adjacency)
    # 初始分數
    regions = evaluate_regions(img_t, segments, regions, model, baseline_y,
                               pred_class, mode=mode, delta=delta, mosaic_block=mosaic_block, batch_size=batch_size)

    
    # jump out if it decreases too much
    # current_best_R = max(regions, key=lambda r: abs(r.score))
    # if abs(current_best_R.score) > 0.20:
    #     print("Early stopping: score too high")
    #     # 結束: 輸出分數最高的區域
    #     best_R = max(regions, key=lambda r: abs(r.score))
    #     final_map = np.zeros((H, W))
    #     for lbl in best_R.labels:
    #         final_map[segments == lbl] = best_R.score
    #     return final_map, pred_class, best_R


    # draw the current best region
    best_R = max(regions, key=lambda r: abs(r.score))
    sens_map_current = np.zeros((H, W))
    for lbl in best_R.labels:
        sens_map_current[segments == lbl] = best_R.score
    draw_current_best_region(img_t, segments, best_R,
                            save_path=os.path.join(f"{SAVE_DIR}/{PERTURB_MODE}/{N_SEGMENTS}",
                                                    f"iter_{0:03d}_region.png"))
    
    Best_Time = 0
    Best_R = best_R
    # 疊代擴展
    for it in range(max_iter):
        
        print(f"Iteration {it+1}/{max_iter}: region count = {len(regions)}")
        candidates = []
        each_regions_candidates = []
        for R in regions:
            each_regions_candidates.append(0)
            if not R.neighbors:
                continue
            # # 嘗試擴展一個隨機鄰居
            # add_lbl = random.choice(list(R.neighbors))
            # new_R = Region(R.labels | {add_lbl})
            # new_R.num_pixels = R.num_pixels + np.sum(segments == add_lbl)
            # new_R.update_neighbors(adjacency)
            # candidates.append(new_R)
            
            # #嘗試刪掉一個在set中，但不是在內部而是外部的 (有跟其他非這集合的label鄰接的)
            # if len(R.labels) > 1:
            #     remove_lbl = random.choice([l for l in R.labels if adjacency[l] - R.labels])
            #     new_labels = R.labels - {remove_lbl}
            #     new_R2 = Region(new_labels)
            #     new_R2.num_pixels = R.num_pixels - np.sum(segments == remove_lbl)
            #     new_R2.update_neighbors(adjacency)
            #     candidates.append(new_R2)
            
            # 嘗試擴展所有鄰居
            for add_lbl in R.neighbors:
                new_R = Region(R.labels | {add_lbl})
                new_R.num_pixels = R.num_pixels + np.sum(segments == add_lbl)
                new_R.update_neighbors(adjacency)
                candidates.append(new_R)
                each_regions_candidates[-1] += 1
            
            # 嘗試刪掉一個在set中，但不是在內部而是外部的 (如果刪掉他，則剩餘的region還是連通的)
            if len(R.labels) > 1:
                for remove_lbl in [l for l in R.labels if connect_graph(R.labels - {l}, adjacency)]:
                    new_labels = R.labels - {remove_lbl}
                    if connect_graph(new_labels, adjacency):
                        new_R2 = Region(new_labels)
                        new_R2.num_pixels = R.num_pixels - np.sum(segments == remove_lbl)
                        new_R2.update_neighbors(adjacency)
                        candidates.append(new_R2)
                        each_regions_candidates[-1] += 1

        if not candidates:
            break

        # 評估候選擴展
        candidates = evaluate_regions(img_t, segments, candidates, model, baseline_y,
                                      pred_class, mode=mode, delta=delta, mosaic_block=mosaic_block, batch_size=batch_size)

        # 接受條件: 若分數提升超過平均增益
        avg_score = np.mean([abs(R.score)/R.num_pixels for R in regions])
        accepted = []
        Region_index = 0
        candidates_index = 0
        for R_count in each_regions_candidates:
            if R_count == 0:
                Region_index += 1
                continue
            old_R = regions[Region_index]
            for _ in range(R_count):
                new_R = candidates[candidates_index]
                if abs(new_R.score)/new_R.num_pixels > abs(old_R.score)/old_R.num_pixels:
                    accepted.append(new_R)
                elif abs(old_R.score)/old_R.num_pixels > avg_score:
                    accepted.append(old_R)
                    accepted.append(new_R)
                else:
                    accepted.append(old_R)
                candidates_index += 1
            Region_index += 1
        
        # for old_R, new_R in zip(regions, candidates):
        #     if abs(new_R.score)/new_R.num_pixels > abs(old_R.score)/old_R.num_pixels:
        #         accepted.append(new_R)
        #     elif abs(old_R.score)/old_R.num_pixels > avg_score:
        #         accepted.append(old_R)
        #         accepted.append(new_R)
        #     else:
        #         accepted.append(old_R)
        # only keep the first 256 best regions
        # 隨著時間演進 只剩越來越少的candidantes
        num_candidates = 256 // ((it + 1) // 2 + 1)
        accepted = sorted(accepted, key=lambda r: abs(r.score), reverse=True)[:num_candidates]
        regions = accepted
        
        # jump out if it decreases too much
        current_best_R = max(regions, key=lambda r: abs(r.score))
        # if abs(current_best_R.score) > 0.20:
        #     print("Early stopping: score too high")
        #     break
        
        if Best_R.labels != current_best_R.labels:
            Best_R = current_best_R
            Best_Time = 0
        else:
            Best_Time += 1
        if Best_Time >= 10:
            print("Early stopping: no improvement for 10 iterations")
            break
        
        # draw the current best region
        # if it % 50 == 0:
        #     best_R = max(regions, key=lambda r: abs(r.score))
        #     sens_map_current = np.zeros((H, W))
        #     for lbl in best_R.labels:
        #         sens_map_current[segments == lbl] = best_R.score
        #     draw_current_best_region(img_t, segments, best_R,
        #                             save_path=os.path.join(f"{SAVE_DIR}/{PERTURB_MODE}/{N_SEGMENTS}",
        #                                                     f"iter_{it+1:03d}_region.png"))
        
        
        # 更新鄰居資訊
        for R in regions:
            R.update_neighbors(adjacency)

        
        
        
    # 結束: 輸出分數最高的區域
    best_R = max(regions, key=lambda r: abs(r.score))
    final_map = np.zeros((H, W))
    for lbl in best_R.labels:
        final_map[segments == lbl] = best_R.score

    return final_map, pred_class, best_R

def draw_current_best_region(img_t, segments, best_R, save_path):
    img_disp = tensor_to_img(img_t)
    H, W = segments.shape
    sens_map_current = np.zeros((H, W))
    for lbl in best_R.labels:
        sens_map_current[segments == lbl] = best_R.score

    fig, ax = plt.subplots(1,5, figsize=(15,5))
    ax[0].imshow(img_disp)
    ax[0].set_title(f"Original")
    ax[0].axis("off")

    colored = label2rgb(segments, img_disp, kind='avg', bg_label=None)
    ax[1].imshow(colored)
    ax[1].set_title(f"Superpixel Segmentation")
    ax[1].axis("off")

    modified_img = apply_superpixel_modification(img_t, segments, best_R.labels,
                                                mode=PERTURB_MODE, delta=DELTA, mosaic_block=MOSAIC_BLOCK)
    modified_img_disp = tensor_to_img(modified_img)
    ax[2].imshow(modified_img_disp)
    ax[2].set_title(f"Modified Image")
    ax[2].axis("off")

    ax[3].imshow(np.abs(sens_map_current), cmap="gray")
    ax[3].set_title(f"|softmax Δ|")
    fig.colorbar(ax[3].images[0], ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].axis("off")

    ax[4].imshow(img_disp)
    im4 = ax[4].imshow(sens_map_current, cmap="seismic", alpha=0.65)
    ax[4].set_title("Current Best Region")
    fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].axis("off")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


perturb_modes = ['mosaic','delta']
n_segments_list = [10,20,50,100,1024]
Delta = 0.0001
MAX_iter_list = [500]
import random
random.seed(91123)
if __name__ == "__main__":
    # random generate index for 30 images
    random_indices = random.sample(range(len(dataset)), 30)
    for perturb_mode in perturb_modes:
        PERTURB_MODE = perturb_mode
        for n_segments in n_segments_list:
            N_SEGMENTS = n_segments
            for MAX_iter in MAX_iter_list:
                process_dataset(index=random_indices)


