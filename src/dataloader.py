import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# 定义数据集
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

    def writer(self):
        writer = SummaryWriter(log_dir='../logs')
        for epoch in range(2):
            step = 0
            for data in self.test_loader:
                imgs, labels = data
                writer.add_images('epoch: test_images{}'.format(epoch), imgs, step)
                step = step + 1

        writer.close()

if __name__ == '__main__':
    my_dataloader = MyDataloader()