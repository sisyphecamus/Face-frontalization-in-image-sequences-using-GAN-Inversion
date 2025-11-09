import sys
sys.path.append('.')

from models.encoders import backbone_encoders
from argparse import Namespace

opts = Namespace(input_nc=3, device='cpu')
encoder = backbone_encoders.EfficientEncoder(50, 'ir_se', opts)

print("EfficientEncoder 的所有子模块:")
for name, module in encoder.named_modules():
    if name:  # 跳过根模块
        print(f"  {name}: {type(module).__name__}")

print("\nEfficientEncoder 的所有参数/缓冲区:")
for name, param in encoder.named_parameters():
    print(f"  {name}: {param.shape}")