import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn.functional as F

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
random_indices = random.sample(range(len(dataset)), 50)
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
                img_batchnorm += F.interpolate(((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))*bn2_alpha[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))+model.bn2.bias[index].to("cpu")).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                img_batchnorm -= F.interpolate(((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))*bn2_alpha[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))+model.bn2.bias[index].to("cpu")).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
        else:
            if bn3_alpha[pred] > 0:
                img_batchnorm -= F.interpolate(((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))*bn2_alpha[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))+model.bn2.bias[index].to("cpu")).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                img_batchnorm += F.interpolate(((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))*bn2_alpha[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))+model.bn2.bias[index].to("cpu")).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]

    
    img_wo_bias = torch.zeros(32,32)
    for index ,fc in enumerate(fc_value_sign):
        if fc > 0:
            if bn2_alpha[index] > 0:
                if bn3_alpha[pred] > 0:
                    img_wo_bias += F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_bias -= F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if bn3_alpha[pred] > 0:
                    img_wo_bias -= F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_bias += F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]

        else:
            if bn2_alpha[index] < 0:
                if bn3_alpha[pred] > 0:
                    img_wo_bias += F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_bias -= F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if bn3_alpha[pred] > 0:
                    img_wo_bias -= F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_bias += F.interpolate((((acts["layer3"][0][index] - model.bn2.running_mean[index].to("cpu"))/((model.bn2.running_var[index].to("cpu")+1e-05)**(1/2)))).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]


    img_wo_batchnorm = torch.zeros(32,32)
    for index ,fc in enumerate(fc_value_sign):
        if fc > 0:
            if bn2_alpha[index] > 0:
                if bn3_alpha[pred] > 0:
                    img_wo_batchnorm += F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_batchnorm -= F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if bn3_alpha[pred] > 0:
                    img_wo_batchnorm -= F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_batchnorm += F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]

        else:
            if bn2_alpha[index] < 0:
                if bn3_alpha[pred] > 0:
                    img_wo_batchnorm += F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_batchnorm -= F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
            else:
                if bn3_alpha[pred] > 0:
                    img_wo_batchnorm -= F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]
                else:
                    img_wo_batchnorm += F.interpolate((acts["layer3"][0][index]).unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False)[0][0]

    plt.figure(figsize=(27,6))
    # original image
    plt.subplot(1, 7, 1)
    plt.title(f"Label: {CIFAR10_CLASSES[label]}\nPred: {CIFAR10_CLASSES[pred]}")
    plt.imshow(tensor_to_img(img_t.cpu()))
    plt.axis("off")
    
    # proposed method image
    plt.subplot(1, 7, 2)
    im3 = plt.imshow(tensor_to_img(img_batchnorm), cmap="seismic")
    plt.title("Proposed Method")
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.axis("off")
    
    # proposed method overlay original img
    plt.subplot(1, 7, 3)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im4 = plt.imshow(tensor_to_img(img_batchnorm), cmap="seismic", alpha=0.5)
    plt.title("Proposed Method Overlay")
    plt.colorbar(im4, fraction=0.046, pad=0.04)
    plt.axis("off")

    # proposed method w/o bias image
    plt.subplot(1, 7, 4)
    im5 = plt.imshow(tensor_to_img(img_wo_bias), cmap="seismic")
    plt.title("Proposed Method w/o Bias")
    plt.colorbar(im5, fraction=0.046, pad=0.04)
    plt.axis("off")
    
    # proposed method w/o bias overlay original img
    plt.subplot(1, 7, 5)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im6 = plt.imshow(tensor_to_img(img_wo_bias), cmap="seismic", alpha=0.5)
    plt.title("Proposed Method w/o Bias Overlay")
    plt.colorbar(im6, fraction=0.046, pad=0.04)
    plt.axis("off")
    
    # proposed method w/o batchnorm image
    plt.subplot(1, 7, 6)
    im7 = plt.imshow(tensor_to_img(img_wo_batchnorm), cmap="seismic")
    plt.title("Proposed Method w/o BatchNorm")  
    plt.colorbar(im7, fraction=0.046, pad=0.04)
    plt.axis("off")
    
    # proposed method w/o batchnorm overlay original img
    plt.subplot(1, 7, 7)
    plt.imshow(tensor_to_img(img_t.cpu()))
    im8 = plt.imshow(tensor_to_img(img_wo_batchnorm), cmap="seismic", alpha=0.5)
    plt.title("Proposed Method w/o BatchNorm Overlay")
    plt.colorbar(im8, fraction=0.046, pad=0.04)
    plt.axis("off")
    

    plt.tight_layout()
    import os 
    if not os.path.exists("bin_gradcam_ablitation"):
        os.makedirs("bin_gradcam_ablitation")
    plt.savefig(f"bin_gradcam_ablitation/{idx}.png", bbox_inches="tight", dpi=150)
    plt.close()


    


