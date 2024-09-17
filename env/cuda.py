import torch

def get_device():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA 可用！可用的 GPU 数量：{num_devices}")

        for device_id in range(num_devices):
            device_name = torch.cuda.get_device_name(device_id)
            print(f"设备id: {device_id}, 设备name: {device_name}")

        # 设置当前设备
        device = torch.device('cuda')
        print(f"当前使用的设备device: {device}")
    else:
        print("CUDA 不可用，使用 CPU。")
        device = torch.device('cpu')
        print(f"当前使用的设备device: {device}")

    return device

if __name__ == "__main__":
    get_device()
