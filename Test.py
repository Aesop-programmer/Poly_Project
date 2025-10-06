
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

# ========= Dataset =========
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**__imagenet_stats)
])
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=False)


# ========= Models =========
model_config = {'input_size': 32, 'dataset': "cifar10"}

# model_bin_w_bn = models.__dict__["resnet_binary"](**model_config)
# ckpt = torch.load("model_best_cifar10_bin_w_bn.pth.tar")
# model_bin_w_bn.load_state_dict(ckpt['state_dict'])

# model_bin_wo_bn = models.__dict__["resnet_binary"](**model_config)
# ckpt = torch.load("model_best_cifar10_bin_wo_bn.pth.tar")
# model_bin_wo_bn.load_state_dict(ckpt['state_dict'])

# model_real = models.__dict__["resnet"](**model_config)
# ckpt = torch.load("model_best_cifar10_real.pth (1).tar")
# model_real.load_state_dict(ckpt['state_dict'])

# model_bin_w_layernorm = models.__dict__["resnet_binary"](**model_config)
# ckpt = torch.load("model_best_cifar10_bin_w_layernorm.pth.tar")
# model_bin_w_layernorm.load_state_dict(ckpt['state_dict'])

model_bin_w_scaling = models.__dict__["resnet_binary"](**model_config)
ckpt = torch.load("model_best_cifar10_bin_w_scaling.pth.tar")
model_bin_w_scaling.load_state_dict(ckpt['state_dict'])

models_to_test = {
    # "Binary w/ BN": model_bin_w_bn,
    # "Binary w/o BN": model_bin_wo_bn,
    # "Real-valued": model_real,
    # "Binary w/ LayerNorm": model_bin_w_layernorm,
    "Binary w/ Scaling": model_bin_w_scaling,
}

# ========= Gradient Analysis =========
def analyze_gradients(model, images, labels):
    model.train()  # 啟用 BN/dropout 訓練模式
    images = images.clone().requires_grad_()

    output = model(images)   # logits
    output.retain_grad()

    loss = F.cross_entropy(output, labels)
    loss.backward()

    return output.detach(), output.grad.detach(), images.grad.detach()

if __name__ == "__main__":
    # 取前一個 batch 測試
    images, labels = next(iter(trainloader))
    print(f"===== Testing on batch size {images.size(0)} =====")

    for name, model in models_to_test.items():
        logits, logits_grad, input_grad = analyze_gradients(model, images, labels)
        print(model(images))
        print(f"\n--- {name} ---")
        print("logits shape:", logits.shape)               # (batch, num_classes)
        print("logits grad shape:", logits_grad.shape)     # (batch, num_classes)
        print(logits_grad)  # logits 對 loss 的梯度