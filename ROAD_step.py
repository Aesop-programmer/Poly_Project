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

drop_cam = []
increase_cam = []
drop_cam_plus = []
increase_cam_plus = []
drop_score_cam = []
increase_score_cam = []
drop_sign_cam = []
increase_sign_cam = []

def draw_auc_curve(step_size, mode,cams ,input_tensor, targets,original_score):
    threshold = []
    for i in range(1,step_size+1):
        threshold.append(int(i*step_size/100))
    scores = [[],[],[],[]]
    for t in threshold:
        if mode == "mr_first":
            road_metric = ROADMostRelevantFirst(percentile=100 - t)
        elif mode == "lr_first":
            road_metric = ROADLeastRelevantFirst(percentile=100 -t)


        for i,cam in enumerate(cams):
            score, _ = road_metric(input_tensor, cam, targets, model, return_visualization=True)
            score = score[0] + original_score
            scores[i].append(score.item())

    return scores
idxs = [1,2,3,4,5,123,5262,5405]
for idx in idxs:
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
    
    cams = [grayscale_cam, grayscale_cam_plus, grayscale_cam_score, grayscale_signcam]
    variants = ["GradCAM", "GradCAM++", "ScoreCAM", "SignCAM"]
    score_mr_first = draw_auc_curve(100, "mr_first", cams, input_tensor, targets, torch.softmax(y, dim=1)[0, pred].detach())
    score_lr_first = draw_auc_curve(100, "lr_first", cams, input_tensor, targets, torch.softmax(y, dim=1)[0, pred].detach())

    import os
    if not os.path.exists(f"ROAD_image/{idx}_{label}_{pred}"):
        os.makedirs(f"ROAD_image/{idx}_{label}_{pred}")

    x_axis = np.linspace(0, 100, len(score_lr_first[0]))  # x 軸 (0~100%)

    plt.figure(figsize=(8, 5))
    for score, variant in zip(score_mr_first, variants):
        auc = np.trapezoid(score, x_axis)  # 計算AUC
        plt.plot(x_axis, score, label=f"{variant} (AUC={auc:.3f})", linewidth=2)

    plt.xlabel("Percentage (%)")
    plt.ylabel("Score")
    plt.title("ROAD Most Relevant First")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/ROAD_Most_Relevant_First.png", bbox_inches="tight", dpi=500)
    plt.close()

    plt.figure(figsize=(8, 5))
    for score, variant in zip(score_lr_first, variants):
        auc = np.trapezoid(score, x_axis)  # 計算AUC
        plt.plot(x_axis, score, label=f"{variant} (AUC={auc:.3f})", linewidth=2)
    plt.xlabel("Percentage (%)")
    plt.ylabel("Score")
    plt.title("ROAD Least Relevant First")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/ROAD_Least_Relevant_First.png", bbox_inches="tight", dpi=500)
    plt.close()
    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # grayscale_cam_plus = grayscale_cam_plus[0, :]
    # grayscale_cam_score = grayscale_cam_score[0, :]


    # cam_scores, cam_visualizations = cam_metric(input_tensor, grayscale_cam, targets, model, return_visualization=True)
    # cam_scores = cam_scores[0]
    # cam_visualizations = cam_visualizations[0]

    # cam_plus_scores, cam_plus_visualizations = cam_metric(input_tensor, grayscale_cam_plus, targets, model, return_visualization=True)
    # cam_plus_scores = cam_plus_scores[0]
    # cam_plus_visualizations = cam_plus_visualizations[0]

    # score_cam_scores, score_cam_visualizations = cam_metric(input_tensor, grayscale_cam_score, targets, model, return_visualization=True)
    # score_cam_scores = score_cam_scores[0]
    # score_cam_visualizations = score_cam_visualizations[0]

    # sign_cam_scores, sign_cam_visualizations = cam_metric(input_tensor, grayscale_signcam, targets, model, return_visualization=True)
    # sign_cam_scores = sign_cam_scores[0]
    # sign_cam_visualizations = sign_cam_visualizations[0]


    # import os
    # if not os.path.exists(f"ROAD_image/{idx}_{label}_{pred}"):
    #     os.makedirs(f"ROAD_image/{idx}_{label}_{pred}")
        
    # # original image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/original.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # # cam image
    # # gradcam image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im4 = plt.imshow(grayscale_cam[0], cmap="seismic", alpha=0.5)
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/gradcam.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # #cam_visualization image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(deprocess_image(cam_visualizations.cpu().numpy().transpose(1,2,0)))
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/gradcam_visualization.png", bbox_inches="tight", dpi=500)
    # plt.close()


    # #gramcam++ image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im5 = plt.imshow(grayscale_cam_plus[0], cmap="seismic", alpha=0.5)
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/gradcam_plus.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # # cam_plus_visualization image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(deprocess_image(cam_plus_visualizations.cpu().numpy().transpose(1,2,0)))
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/gradcam_plus_visualization.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # #scorecam image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im6 = plt.imshow(grayscale_cam_score[0], cmap="seismic", alpha=0.5)
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/scorecam.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # # cam_score_visualization image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(deprocess_image(score_cam_visualizations.cpu().numpy().transpose(1,2,0)))
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/scorecam_visualization.png", bbox_inches="tight", dpi=500)
    # plt.close()
    
    # # signcam image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(tensor_to_img(img_t.cpu()))
    # im7 = plt.imshow(grayscale_signcam[0], cmap="seismic", alpha=0.5)
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/signcam.png", bbox_inches="tight", dpi=500)
    # plt.close()

    # # # cam_sign_visualization image
    # plt.figure(figsize=(6,6))
    # plt.subplot(1, 1, 1)
    # plt.imshow(deprocess_image(sign_cam_visualizations.cpu().numpy().transpose(1,2,0)))
    # plt.axis("off")
    # plt.savefig(f"ROAD_image/{idx}_{label}_{pred}/signcam_visualization.png", bbox_inches="tight", dpi=500)
    # plt.close()