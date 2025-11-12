import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === 3. Dataset (CIFAR-10) ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

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
    return t


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



# === 載入 Dataset ===
transform = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
def get_all_activations(model, x):
    acts = {}
    hooks = []
    def save_output(name):
        def hook(module, inp, out):
            acts[name] = out.detach().cpu()
        return hook
    for name, m in model.named_modules():
        if name in ["tanh2","layer3"]:
            hooks.append(m.register_forward_hook(save_output(name)))
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    return acts

import random 
# random sample 30 images from test set
target_layers = [model.layer3]
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

cam_plus = GradCAMPlusPlus(model=model, target_layers=target_layers)

score_cam = ScoreCAM(model=model, target_layers=target_layers)

random_indices = random.sample(range(len(dataset)), 50)
random_indices = [82,361,722,999,1124,1306,1324,1836,2346,2461,2490,2504,2829,2955,3203,3715,3775,3817,4167,4190,4272,4589,4689,4747,4780,5028
                  ,5123,5238,5262,5266,5405,5706,5716,6287,6379,6382,6603,6815,6917,6977,7808,8015,8268,8353,8653,8881,9019
                  ,9390,9727,9805]
for idx in random_indices:
    img_t, label = dataset[idx]
    x = normalize(img_t).unsqueeze(0).to(device)

    acts = get_all_activations(model, x)
    y = model(x)
    pred = torch.argmax(y)
    fc_value = model.fc.weight[pred]
    fc_value_sign = torch.sign(fc_value)
    bn2_alpha = model.bn2.weight
    bn3_alpha = model.bn3.weight
    img_batchnorm = torch.zeros(32,32)
    for index ,fc in enumerate(fc_value_sign):
        if fc > 0:
            if bn3_alpha[pred] > 0:
                if acts["tanh2"][0][index] > 0:
                    img_batchnorm += F.interpolate((acts["layer3"][0][index] ).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if acts["tanh2"][0][index] < 0:
                    img_batchnorm -= F.interpolate((acts["layer3"][0][index] ).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                
        else:
            if bn3_alpha[pred] > 0:
                if acts["tanh2"][0][index] < 0:
                    img_batchnorm -= F.interpolate((acts["layer3"][0][index] ).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if acts["tanh2"][0][index] > 0:
                    img_batchnorm += F.interpolate((acts["layer3"][0][index] ).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]

    from captum.attr import LayerGradCam
    from torchvision import datasets, transforms
    from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
    from captum.attr import visualization as viz
    from captum.attr import LayerGradCam
    def clamp_attr(attr):
        attr = attr.squeeze().cpu().detach().numpy()
        threshold = np.percentile(np.abs(attr), 95)
        attr = np.clip(attr, -threshold, threshold)
        
        return attr
    
    # Sailency
    saliency = Saliency(model)
    attr_saliency = saliency.attribute(x, target=pred)
    attr_saliency = F.interpolate(attr_saliency, size=(32, 32), mode='bilinear', align_corners=False)
    attr_saliency = torch.tensor(attr_saliency.squeeze().cpu().detach().clone().numpy())
    # attr_saliency = clamp_attr(attr_saliency)
    # attr_saliency = torch.tensor(attr_saliency)
    attr_saliency = attr_saliency.abs().max(dim=0)[0]
    # SmoothGrad
    smoothgrad = NoiseTunnel(saliency)
    attr_smoothgrad = smoothgrad.attribute(x, nt_type="smoothgrad", nt_samples=500, stdevs=0.1, target=pred)
    attr_smoothgrad = F.interpolate(attr_smoothgrad, size=(32, 32), mode='bilinear', align_corners=False)
    attr_smoothgrad = torch.tensor(attr_smoothgrad.squeeze().cpu().detach().clone().numpy())
    # attr_smoothgrad = clamp_attr(attr_smoothgrad)
    # attr_smoothgrad = torch.tensor(attr_smoothgrad)
    attr_smoothgrad = attr_smoothgrad.abs().max(dim=0)[0]
    # Integrated Gradients 
    ig = IntegratedGradients(model)
    attr_ig = ig.attribute(x, target=label, n_steps=300)
    attr_ig = F.interpolate(attr_ig, size=(32, 32), mode='bilinear', align_corners=False)
    attr_ig = torch.tensor(attr_ig.squeeze().cpu().detach().clone().numpy())
    # attr_ig = clamp_attr(attr_ig)
    # attr_ig = torch.tensor(attr_ig)    
    attr_ig = attr_ig.abs().max(dim=0)[0]
    
    
    input_tensor = x # Create an input tensor image for your model..
    targets = [ClassifierOutputTarget(pred)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam_plus = cam_plus(input_tensor=input_tensor, targets=targets)
    grayscale_cam_score = score_cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam_plus = grayscale_cam_plus[0, :]
    grayscale_cam_score = grayscale_cam_score[0, :]
    
    
    # draw one by one
    import os
    if not os.path.exists(f"experiments/{idx}_{label}_{pred}"):
        os.makedirs(f"experiments/{idx}_{label}_{pred}")
        
    # original image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    plt.imshow(tensor_to_img(img_t.cpu()))
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/original.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # saliency map
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    im1 = plt.imshow(tensor_to_img(attr_saliency), cmap="gray")
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/saliency.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # smoothgrad map
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    im2 = plt.imshow(tensor_to_img(attr_smoothgrad), cmap="gray")
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/smoothgrad.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # integrated gradients map
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    im3 = plt.imshow(tensor_to_img(attr_ig), cmap="gray")
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/integrated_gradients.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # gradcam image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im4 = plt.imshow(grayscale_cam, cmap="seismic", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/gradcam.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # gramcam++ image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im5 = plt.imshow(grayscale_cam_plus, cmap="seismic", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/gradcam_plus.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # scorecam image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im6 = plt.imshow(grayscale_cam_score, cmap="seismic", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/scorecam.png", bbox_inches="tight", dpi=500)
    plt.close()
    
    # proposed method image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 1, 1)
    plt.imshow(tensor_to_img(img_t.cpu()))
    plt.imshow(tensor_to_img(img_batchnorm), cmap="seismic", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"experiments/{idx}_{label}_{pred}/proposed_method.png", bbox_inches="tight", dpi=500)
    plt.close()

    # pred_top3_prob = torch.topk(torch.softmax(y, dim=1), k=3).values.squeeze().cpu().detach().numpy()
    # pred_top3_idx = torch.topk(torch.softmax(y, dim=1), k=3).indices.squeeze().cpu().detach().numpy()
    # plt.figure(figsize=(15,6))
    # # original image
    # plt.subplot(1, 5, 1)
    # plt.title(f"Label: {CIFAR10_CLASSES[label]}\nPred: {CIFAR10_CLASSES[pred]}\nTop3: {CIFAR10_CLASSES[pred_top3_idx[0]]}({pred_top3_prob[0]:.2f}), {CIFAR10_CLASSES[pred_top3_idx[1]]}({pred_top3_prob[1]:.2f}), {CIFAR10_CLASSES[pred_top3_idx[2]]}({pred_top3_prob[2]:.2f})")
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # plt.axis("off")
    
    # # gradcam image
    # plt.subplot(1, 5, 2)
    # im1 = plt.imshow(tensor_to_img(attr_gc_bin_w_bn), cmap="seismic")
    # plt.colorbar(im1, fraction=0.046, pad=0.04)
    # plt.title("GradCAM")
    # plt.axis("off")
    
    # #gradcam overlay original img
    # plt.subplot(1, 5, 3)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im2 = plt.imshow(tensor_to_img(attr_gc_bin_w_bn), cmap="seismic", alpha=0.5)
    # plt.colorbar(im2, fraction=0.046, pad=0.04)
    # plt.title("GradCAM Overlay")
    # plt.axis("off")

    # # proposed method image
    # plt.subplot(1, 5, 4)
    # im3 = plt.imshow(tensor_to_img(img_batchnorm), cmap="seismic")
    # plt.title("Proposed Method")
    # plt.colorbar(im3, fraction=0.046, pad=0.04)
    # plt.axis("off")
    
    # # proposed method overlay original img
    # plt.subplot(1, 5, 5)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im4 = plt.imshow(tensor_to_img(img_batchnorm), cmap="seismic", alpha=0.5)
    # plt.title("Proposed Method Overlay")
    # plt.colorbar(im4, fraction=0.046, pad=0.04)
    # plt.axis("off")

    # plt.tight_layout()
    # import os 
    # if not os.path.exists("bin_gradcam_vs_proposed_v2_relu"):
    #     os.makedirs("bin_gradcam_vs_proposed_v2_relu")
    # plt.savefig(f"bin_gradcam_vs_proposed_v2_relu/{idx}.png", bbox_inches="tight", dpi=150)
    # plt.close()
    # plt.figure(figsize=(9,6))
    # # original image
    # plt.subplot(1, 2, 1)
    # plt.title(f"Label: {CIFAR10_CLASSES[label]}, Pred: {CIFAR10_CLASSES[pred]}\n",fontdict={'fontsize': 20})
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # plt.axis("off")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im2 = plt.imshow(tensor_to_img(attr_gc_bin_w_bn), cmap="seismic", alpha=0.5)
    # plt.colorbar(im2, fraction=0.046, pad=0.04)
    # plt.title("Grad-CAM", fontdict={'fontsize': 20})
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(f"bin_gradcam_vs_proposed_v2_relu/{idx}_gradcam.png", bbox_inches="tight", dpi=500)
    
    

    


