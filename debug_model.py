import torch

checkpoint = torch.load(r'D:\Amit Data\Amit data\Foot Deformities Detection\Model\monai_densenet_efficient.pth', map_location='cpu', weights_only=False)
print(f"Type: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"Keys: {list(checkpoint.keys())}")
    for key, value in checkpoint.items():
        print(f"{key}: {type(value)}")