import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class SinDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集。

        参数：
        - data: 形状为 [[tau个样本数, 第tau+1的label],...] 的 numpy 数组。
        """
        self.data = data
        self.features = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(data[:, -1], dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。

        返回：
        - feature: 前 tau 个点的值，形状为 [tau]。
        - label: 当前点的值，标量。
        """
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# 生成训练集和测试集的 DataLoader
def get_sin_dataloaders(tau, num_samples, batch_size, rate=0.8, sample_range=(0, 4*np.pi), pure_data=False):
    """
    生成用于训练和测试的 DataLoader。

    参数：
    - tau: 序列长度，使用前 tau 个点预测下一个点。
    - num_samples: 生成的数据点总数。
    - batch_size: DataLoader 中每个批次的大小。
    - rate: 训练集所占比例，默认为 0.8。

    返回：
    - train_loader: 训练集的 DataLoader。
    - test_loader: 测试集的 DataLoader。
    """
    # 生成数据
    data = generate_sin_data(tau, num_samples, sample_range, pure_data)
    
    # 划分训练集和测试集
    num_train = int(len(data) * rate)
    train_data = data[:num_train]
    test_data = data[num_train:]
    
    # 创建数据集
    train_dataset = SinDataset(train_data)
    test_dataset = SinDataset(test_data)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # debug方便
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# 生成正弦函数数据
def generate_sin_data(seq_length, num_samples, datarange=(0, 4*np.pi), pure_data=False):
    x = np.linspace(datarange[0], datarange[1], num_samples)
    if pure_data:
        y = np.sin(x)
    else:
        y = np.sin(x) + 0.1 * np.random.randn(num_samples)
    data = []
    for i in range(len(y) - seq_length):
        data.append(y[i:i+seq_length+1])
    data = np.array(data)
    return data


# 可视化函数
def visualize_sin_data(features, labels, num_show=5):
    img_height = 400
    img_width = 600
    for i in range(num_show):
        feature = features[i]
        label = labels[i]
        seq = np.concatenate((feature, [label]))
        seq_len = len(seq)
        
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        x_coords = np.linspace(50, img_width - 50, seq_len)
        y_min = seq.min()
        y_max = seq.max()
        y_coords = (seq - y_min) / (y_max - y_min + 1e-8)
        y_coords = img_height - (img_height - 100) * y_coords - 50
        y_coords = y_coords.astype(np.int32)
        
        # 绘制特征点（蓝色）
        for j in range(len(feature)):
            x = int(x_coords[j])
            y = int(y_coords[j])
            cv2.circle(img, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
        
        # 绘制标签点（红色）
        x = int(x_coords[-1])
        y = int(y_coords[-1])
        cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        
        # 连接特征点的线（蓝色）
        for j in range(len(feature) - 1):
            x1 = int(x_coords[j])
            y1 = int(y_coords[j])
            x2 = int(x_coords[j + 1])
            y2 = int(y_coords[j + 1])
            cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
        
        # 连接最后一个特征点和标签点（绿色）
        x1 = int(x_coords[-2])
        y1 = int(y_coords[-2])
        x2 = int(x_coords[-1])
        y2 = int(y_coords[-1])
        cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        
        # 添加说明文字
        cv2.putText(img, f"Sample {i+1}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(img, "Features (Blue), Label (Red)", (50, img_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        # 显示图像
        cv2.imshow(f"Sample {i+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    seq_length = 5  # tau
    num_samples = 1000
    train_loader, test_loader = get_sin_dataloaders(seq_length, num_samples, batch_size=32)
    features, labels = next(iter(train_loader))
    visualize_sin_data(features, labels, num_show=5)
