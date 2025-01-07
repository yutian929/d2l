import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead

# 1. 定义自定义数据集类
class BananaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_file = os.path.join(root_dir, 'label.csv')
        self.annotations = pd.read_csv(self.annotations_file)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label = self.annotations.iloc[idx, 1]
        xmin = self.annotations.iloc[idx, 2]
        ymin = self.annotations.iloc[idx, 3]
        xmax = self.annotations.iloc[idx, 4]
        ymax = self.annotations.iloc[idx, 5]
        
        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.as_tensor([label + 1], dtype=torch.int64)  # 类别从1开始，0为背景
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# 2. 定义数据变换
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 创建数据加载器
train_root = '/mnt/zhangyutian/projects/finished/task10_yutian/d2l/d2l/datasets/detection/banana-detection/bananas_train'
val_root = '/mnt/zhangyutian/projects/finished/task10_yutian/d2l/d2l/datasets/detection/banana-detection/bananas_val'

train_dataset = BananaDataset(root_dir=train_root, transform=train_transform)
val_dataset = BananaDataset(root_dir=val_root, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=lambda x: list(zip(*x)))

# 4. 加载预训练的SSD模型并调整
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

# 获取in_channels
in_channels = [module.in_channels for module in model.head.classification_head.module_list]
num_anchors = model.anchor_generator.num_anchors_per_location()

# 重新定义头模型
num_classes = 2  # 背景和香蕉
model.head = SSDHead(
    in_channels,
    num_anchors,
    num_classes
)

# 5. 定义优化器和学习率调度器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 6. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
    
    lr_scheduler.step()
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f'ssd_model_epoch_{epoch+1}.pth')

# 7. 预测
def predict(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions

# 示例预测
image_path = '/mnt/zhangyutian/projects/finished/task10_yutian/d2l/d2l/datasets/detection/banana-detection/bananas_val/images/0.png'
predictions = predict(image_path, model, val_transform)
print(predictions)
# img_name,label,xmin,ymin,xmax,ymax
# 0.png,0,183,63,241,112