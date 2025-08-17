from PIL import Image
import cv2
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

class MyTransform:
    def ToTensor(self):
        img_path = 'dataset/train/ants_image/0013035.jpg'
        # PIL to tensor
        img = Image.open(img_path)
        tensor_img = transforms.ToTensor()(img)

        # ndarray to tensor
        cv_img = cv2.imread(img_path)
        tensor_cv_img = transforms.ToTensor()(cv_img)

        writer = SummaryWriter(log_dir='logs')
        writer.add_image('tensor_ant_img', tensor_img)

        writer.close()

    def Normalize(self):
        writer = SummaryWriter(log_dir='logs')
        img = Image.open("images/lenna.png")

        tensor_img = transforms.ToTensor()(img)
        writer.add_image('normalize_img', tensor_img, 1)

        normalize_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tensor_img)
        writer.add_image('normalize_img', normalize_img, 2)

    def Resize(self):
        writer = SummaryWriter(log_dir='logs')
        img = Image.open("images/lenna.png")

        img_resize = transforms.Resize((512, 512))(img)
        tensor_img = transforms.ToTensor()(img_resize)
        writer.add_image('resize_img', tensor_img, 1)

    def Compose(self):
        img = Image.open("images/lenna.png")
        transform = transforms.Compose([
            transforms.Resize((5, 5)),
            transforms.ToTensor(),
        ])
        tensor_img = transform(img)
        print(tensor_img)

class MyDataset:
    def Dataset(self):
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
        test_set = datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)
        print(train_set)
        print(test_set)

        writer = SummaryWriter(log_dir='logs')
        for i in range(10):
            img, label = train_set[i]
            writer.add_image('train_set', img, i)

        for i in range(10):
            img, label = test_set[i]
            writer.add_image('test_set', img, i)
        writer.close()

if __name__ == '__main__':
    my_transform = MyTransform()
    # my_transform.ToTensor()
    # my_transform.Normalize()
    # my_transform.Resize()
    # my_transform.Compose()
    my_dataset = MyDataset()
    my_dataset.Dataset()
