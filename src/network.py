import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter(log_dir='../logs/network')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CIFAR10_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class vgg16_model:
    def __init__(self):
        super().__init__()
        self.vgg16_false = torchvision.models.vgg16(pretrained=False)
        self.vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

    def modify(self):
        self.vgg16_false.classifier[6] = nn.Linear(4096, 10)
        self.vgg16_true.classifier.add_module('7', nn.Linear(1000, 10))


class MyDataloader(object):

    def __init__(self):
        dataset_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_set = datasets.CIFAR10(root='../dataset', train=True, transform=dataset_transform, download=True)
        test_set = datasets.CIFAR10(root='../dataset', train=False, transform=dataset_transform, download=True)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )


def train(model, dataloader, epochs=5):
    model.train()
    # 随机梯度下降优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for epoch in range(epochs):
        running_loss = 0.0
        for data in dataloader.train_loader:
            imgs, targets = data
            # 将数据移动到GPU
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            # 通过反向传播与优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 损失函数累加
            running_loss += loss.item()
        print(f"epoch: {epoch + 1}, train_loss: {running_loss / len(dataloader.train_loader)}")
        writer.add_scalar('train_loss', running_loss / len(dataloader.train_loader), epoch + 1)
        # 测试
        test(model, dataloader, epoch)
        save_model(model, f"../models/cifar10_model_{epoch + 1}.pth")


def test(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader.test_loader:
            imgs, targets = data
            # 将数据移动到GPU
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            accuracy = (outputs.argmax(dim=1) == targets).sum()
            total_loss += loss.item()
            total_accuracy += accuracy
    writer.add_scalar('test_loss', total_loss / len(dataloader.test_loader), epoch + 1)
    writer.add_scalar('test_accuracy', 100 * total_accuracy / len(dataloader.test_loader.dataset), epoch + 1)
    print(
        f"test_loss: {total_loss / len(dataloader.test_loader)}, test_accuracy: {100 * total_accuracy / len(dataloader.test_loader.dataset)}%")


def save_model(model, path):
    # 1.保存模型结构+模型参数
    # torch.save(models, path)
    # 2.保存模型参数
    torch.save(model.state_dict(), path)


# 加载模型函数
def load_model(model, path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.加载模型结构+模型参数
        # models = torch.load(path)
        # 2.加载模型参数
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"模型加载成功！device={device}")
    except Exception as e:
        print(f"模型加载失败: {e}")


def cifar10_model_train():
    # 创建模型
    model = CIFAR10_model().to(device)
    # 加载数据
    dataloader = MyDataloader()
    # 训练
    train(model, dataloader, epochs=20)
    print("train complete")


if __name__ == '__main__':

    print(f"current device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    start_time = time.time()
    cifar10_model_train()
    end_time = time.time()
    print(f"time: {end_time - start_time}")

    # cifar10_model_train()

    # input = torch.randn(64, 3, 32, 32)
    # output = models(input)
    # writer.add_graph(models, input)
    # writer.close()
    # print(models)
