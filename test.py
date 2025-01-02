import torch

if torch.cuda.is_available():
    print("CUDA可用")
    print(f"当前CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用")