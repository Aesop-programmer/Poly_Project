from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
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
target_layers = [model.layer3]
# Note: input_tensor can be a batch tensor with several images!




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

def SignCAM(model,input_tensor,target):
    
    acts = get_all_activations(model, input_tensor)
    pred = torch.argmax(target)
    fc_value = model.fc.weight[pred]
    fc_value_sign = torch.sign(fc_value)
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
    return img_batchnorm.unsqueeze(0).numpy()

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

cam_plus = GradCAMPlusPlus(model=model, target_layers=target_layers)

score_cam = ScoreCAM(model=model, target_layers=target_layers)
score_cam.batch_size = 512
cam_metric = ROADLeastRelevantFirst(percentile=50)

scorecam_area = []
signcam_area = []


def drop_in_confidence(drop_score, pred, y, model):
    return max(0,-drop_score)/torch.softmax(y, dim=1)[0, pred].detach()

def increase_in_confidence(drop_score, pred, y, model):
    if drop_score > 0:
        return 1.0
    else:
        return 0.0

for idx in range(len(dataset)):
    img_t, label = dataset[idx]
    x = normalize(img_t).unsqueeze(0).to(device)
    y = model(x)
    pred = torch.argmax(y)

    input_tensor = x # Create an input tensor image for your model..

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.




    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    targets = [ClassifierOutputSoftmaxTarget(pred)]
    grayscale_cam_score = score_cam(input_tensor=input_tensor, targets=targets)
    
    grayscale_signcam = SignCAM(model, input_tensor, pred)
    
    
    #calculate the how many pixel in heatmap are non-zero (threshold=10^-6)
    scorecam_area.append((grayscale_cam_score[0] > grayscale_cam_score[0].mean()).sum().item())
    signcam_area.append((grayscale_signcam[0] > grayscale_signcam[0].mean()).sum().item())

print("ScoreCAM average area:", np.mean(scorecam_area))
print("SignCAM average area:", np.mean(signcam_area))