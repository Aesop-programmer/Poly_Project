
import numpy as np
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
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
    model_bin_w_bn = models.__dict__["resnet_binary"]
    model_config = {'input_size': 32, 'dataset': "cifar10"}
    model_bin_w_bn = model_bin_w_bn(**model_config)
    checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/results/2025-09-23_11-49-02/checkpoint.pth.tar")
    model_bin_w_bn.load_state_dict(checkpoint['state_dict'])
    model_bin_w_bn.eval()

    model_bin_wo_bn = models.__dict__["resnet_binary"]
    model_bin_wo_bn = model_bin_wo_bn(**model_config)
    checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_bin_wo_bn.pth.tar")
    model_bin_wo_bn.load_state_dict(checkpoint['state_dict'])
    model_bin_wo_bn.eval()

    model_real = models.__dict__["resnet"]
    model_real = model_real(**model_config)
    checkpoint = torch.load("C:/Users/abc09/Desktop/master/蒙特婁理工大學實習/Poly_Project/model_best_cifar10_real.pth (1).tar")
    model_real.load_state_dict(checkpoint['state_dict'])
    model_real.eval()
    
    
    #test on testset
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    acc_bin_w_bn = test(model_bin_w_bn, testloader)
    acc_bin_wo_bn = test(model_bin_wo_bn, testloader)
    acc_real = test(model_real, testloader)
    print(f"Test Accuracy of Real-valued ResNet: {acc_real:.2f}%")
    print(f"Test Accuracy of Binary ResNet with BN: {acc_bin_w_bn:.2f}%")
    print(f"Test Accuracy of Binary ResNet without BN: {acc_bin_wo_bn:.2f}%")