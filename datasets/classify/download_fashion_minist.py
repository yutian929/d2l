import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torchvision

def get_fashion_mnist_loaders(data_dir, batch_size=64, num_workers=2, resize_size=None):
    """
    获取 Fashion-MNIST 数据集的训练和测试 DataLoader。

    参数：
    - data_dir: 数据集存储路径。
    - batch_size: 批量大小。
    - num_workers: 加载数据的子进程数。
    - resize_size: 图像调整后的尺寸，格式为 (H, W)，例如 (224, 224)。如果为 None，则不调整尺寸。

    返回：
    - train_loader: 训练集 DataLoader。
    - test_loader: 测试集 DataLoader。
    """

    # 如果路径不存在，创建该路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 定义数据集的归一化转换
    transform_list = []

    # 如果指定了调整尺寸，添加调整尺寸的转换
    if resize_size is not None:
        transform_list.append(transforms.Resize(resize_size))

    # 添加转换为张量和标准化的转换
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将像素值标准化到 [-1, 1]
    ])

    transform = transforms.Compose(transform_list)

    # 下载并加载训练集和测试集
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 为了debug方便以及产生对比，将 shuffle 设置为 False
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

def show_batch_images(images, labels, classes, resize_size=None):
    """
    显示一个批次的图像，并在每张图片上添加标签。

    参数：
    - images: 图像张量，形状为 [B, C, H, W]。
    - labels: 标签张量，形状为 [B]。
    - classes: 类别名称列表。
    - resize_size: 图像调整后的尺寸，格式为 (H, W)，例如 (224, 224)。如果为 None，则使用原始尺寸。
    """
    # 将图片张量组合成网格
    img_per_row = 8  # 每行显示的图片数量

    # 如果指定了调整尺寸，计算网格尺寸
    if resize_size is not None:
        grid_img = torchvision.utils.make_grid(images, nrow=img_per_row, padding=2)
    else:
        grid_img = torchvision.utils.make_grid(images, nrow=img_per_row, padding=2)

    # 将张量转换为 NumPy 数组
    np_img = grid_img.numpy()
    # 反标准化
    np_img = np.transpose(np_img, (1, 2, 0)) * 0.5 + 0.5  # [H, W, C]
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    # 将 RGB 转换为 BGR（OpenCV 使用 BGR）
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # 计算每张图片的宽度和高度
    batch_size = images.size(0)
    grid_height, grid_width = np_img.shape[:2]

    # 计算每张图片的高度和宽度
    num_rows = (batch_size + img_per_row - 1) // img_per_row  # 总行数
    img_height = grid_height // num_rows
    img_width = grid_width // img_per_row

    # 在每张图片上添加标签
    for idx in range(batch_size):
        # 计算图片的位置
        row = idx // img_per_row
        col = idx % img_per_row
        x = col * img_width
        y = row * img_height

        # 获取标签名称
        label = classes[labels[idx]]

        # 在图片上添加文本
        cv2.putText(
            np_img,
            label[:],  # 只取前两个字符，避免文本过长
            (x + 5, y + 15),  # 文本位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # 字体大小
            (0, 255, 0),  # 字体颜色 (B, G, R)
            1  # 字体粗细
        )

    # 显示图片
    cv2.imshow('Batch Images', np_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例：使用该函数获取 DataLoader
if __name__ == '__main__':
    data_dir = '/home/yutian/projects/d2l/d2l/datasets/classify'
    batch_size = 64  # 可根据需要调整
    num_workers = 2  # 根据你的CPU核心数调整
    resize_size = (224, 224)  # 调整图像尺寸，例如 (224, 224)。如果不需要调整，设置为 None

    train_loader, test_loader = get_fashion_mnist_loaders(data_dir, batch_size, num_workers, resize_size)

    # 定义标签类别
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 获取一个批次的数据
    images, labels = next(iter(train_loader))

    # 显示批次图像
    show_batch_images(images, labels, classes, resize_size)
