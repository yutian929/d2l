import os
import requests
import zipfile

# 下载数据集
url = "https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip"
data_dir = "./HotdogData/"
os.makedirs(data_dir, exist_ok=True)
zip_path = os.path.join(data_dir, "hotdog.zip")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("Downloading hotdog dataset...")
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)

# 解压数据集
print(f"Extracting hotdog dataset {zip_path}")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

print("Dataset downloaded and extracted.")
