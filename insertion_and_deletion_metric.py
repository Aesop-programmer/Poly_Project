import models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import conv2d

from RISE.evaluation import CausalMetric, auc, gkern

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
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

target_layers = [model.layer3]
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

cam_plus = GradCAMPlusPlus(model=model, target_layers=target_layers)

score_cam = ScoreCAM(model=model, target_layers=target_layers)
score_cam.batch_size = 512



# insertion and deletion metric
klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

insertion = CausalMetric(model, 'ins', 20, substrate_fn=blur)
deletion = CausalMetric(model, 'del', 20, substrate_fn=torch.zeros_like)


# 隨機sample 1000張圖片來計算 insertion & deletion score
import random
import time
start = time.time()
# idxs = random.sample(range(len(dataset)), 2000)
del_scores = {"cam": [], "cam++": [], "scorecam": [], "signcam": []}
ins_scores = {"cam": [], "cam++": [], "scorecam": [], "signcam": []}
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



    targets = [ClassifierOutputSoftmaxTarget(pred)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam_plus = cam_plus(input_tensor=input_tensor, targets=targets)
    grayscale_cam_score = score_cam(input_tensor=input_tensor, targets=targets)
    
    grayscale_signcam = SignCAM(model, input_tensor, pred)
    
    # import os 
    # if not os.path.exists(f'RISE_image/{idx}'):
    #     os.makedirs(f'RISE_image/{idx}')
    # # original image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # plt.axis("off")
    # plt.savefig(f"./RISE_image/{idx}/original.png", bbox_inches="tight", dpi=500)
    # plt.close()
    # for cam_type, name in zip([grayscale_cam, grayscale_cam_plus, grayscale_cam_score, grayscale_signcam],
    #                      ['gradcam', 'gradcam++', 'scorecam', 'signcam']):
    #     h = deletion.single_run(input_tensor.to("cpu"), cam_type, verbose=1,save_to=f'./RISE_image/{idx}/{name}_deletion.jpg')
    #     h = insertion.single_run(input_tensor.to("cpu"), cam_type, verbose=1,save_to=f'./RISE_image/{idx}/{name}_insertion.jpg')
    cam_list = [grayscale_cam, grayscale_cam_plus, grayscale_cam_score, grayscale_signcam]
    for cam_type, name in zip(cam_list, ['cam', 'cam++', 'scorecam', 'signcam']):
        exp = cam_type

        h = deletion.evaluate(input_tensor.to("cpu"), exp, 1)
        del_scores[name].append(auc(h.mean(1)))

        h = insertion.evaluate(input_tensor.to("cpu"), exp, 1)
        ins_scores[name].append(auc(h.mean(1)))

end = time.time()
print(f"Time taken: {end - start} seconds")
print("Deletion scores:")
for key in del_scores.keys():
    print(f"{key}: {np.mean(del_scores[key])}")

print("Insertion scores:")
for key in ins_scores.keys():
    print(f"{key}: {np.mean(ins_scores[key])}")





