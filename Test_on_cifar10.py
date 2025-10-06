
import numpy as np
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    statistics = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            statistics.append(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    statistics = torch.cat(statistics, dim=0)
    mean = statistics.mean(dim=0)
    std = statistics.std(dim=0)
    return accuracy, mean, std
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import models
if __name__ == "__main__":


    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
    # === 1) Data ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # === 2) Model ===
    # model_bin_w_bn = models.__dict__["resnet_binary"]
    model_config = {'input_size': 32, 'dataset': "cifar10"}
    # model_bin_w_bn = model_bin_w_bn(**model_config)
    # checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_bn.pth.tar")
    # model_bin_w_bn.load_state_dict(checkpoint['state_dict'])
    # model_bin_w_bn.eval()

    # model_bin_wo_bn = models.__dict__["resnet_binary"]
    # model_bin_wo_bn = model_bin_wo_bn(**model_config)
    # checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_wo_bn.pth.tar")
    # model_bin_wo_bn.load_state_dict(checkpoint['state_dict'])
    # model_bin_wo_bn.eval()

    # model_real = models.__dict__["resnet"]
    # model_real = model_real(**model_config)
    # checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar")
    # model_real.load_state_dict(checkpoint['state_dict'])
    # model_real.eval()
    
    # model_bin_w_scaling = models.__dict__["resnet_binary"]
    # model_bin_w_scaling = model_bin_w_scaling(**model_config)
    # checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_w_scaling.pth.tar")
    # model_bin_w_scaling.load_state_dict(checkpoint['state_dict'])
    # model_bin_w_scaling.eval()

    model_bin_w_layernorm = models.__dict__["resnet_binary"]
    model_config = {'input_size': 32, 'dataset': "cifar10"}
    model_bin_w_layernorm = model_bin_w_layernorm(**model_config)
    checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/results/2025-10-02_21-20-40/model_best.pth.tar")
    model_bin_w_layernorm.load_state_dict(checkpoint['state_dict'])
    model_bin_w_layernorm.eval()

    #test on testset
    testloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)
    # acc_bin_w_bn, mean_bin_w_bn, std_bin_w_bn = test(model_bin_w_bn, testloader)
    # print(mean_bin_w_bn, std_bin_w_bn)
    # acc_bin_wo_bn, mean_bin_wo_bn, std_bin_wo_bn = test(model_bin_wo_bn, testloader)
    # print(mean_bin_wo_bn, std_bin_wo_bn)
    # acc_real = test(model_real, testloader)
    # acc_bin_w_scaling, mean_bin_w_scaling, std_bin_w_scaling = test(model_bin_w_scaling, testloader)
    # print(acc_bin_w_scaling)
    # print(mean_bin_w_scaling, std_bin_w_scaling)
    acc_bin_w_layernorm, mean_bin_w_layernorm, std_bin_w_layernorm = test(model_bin_w_layernorm, testloader)
    print(acc_bin_w_layernorm)
    print(mean_bin_w_layernorm, std_bin_w_layernorm)
    # print(f"Test Accuracy of Real-valued ResNet: {acc_real:.2f}%")
    # print(f"Test Accuracy of Binary ResNet with BN: {acc_bin_w_bn:.2f}%")
    # print(f"Test Accuracy of Binary ResNet without BN: {acc_bin_wo_bn:.2f}%")
    # print(f"Test Accuracy of Binary ResNet with Weight Scaling: {acc_bin_w_scaling:.2f}%")
    # print(f"Test Accuracy of Binary ResNet with Layer Normalization: {acc_bin_w_layernorm:.2f}%")