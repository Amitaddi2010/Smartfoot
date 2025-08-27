import torch

try:
    checkpoint = torch.load(r'D:\Amit Data\Amit data\Foot Deformities Detection\Model\monai_densenet_efficient.pth', 
                           map_location='cpu', weights_only=False)
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())}")
        for key, value in checkpoint.items():
            print(f"{key}: {type(value)}")
            if hasattr(value, 'keys') and callable(getattr(value, 'keys')):
                print(f"  - Sub-keys: {list(value.keys())[:5]}...")  # First 5 keys
    else:
        print("Direct model object")
        
except Exception as e:
    print(f"Error: {e}")