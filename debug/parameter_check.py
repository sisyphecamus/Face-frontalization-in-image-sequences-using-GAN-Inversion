import torch
import sys
sys.path.append('.')

from models.face_frontalizer import get_keys
from models.encoders import backbone_encoders
from argparse import Namespace

# 加载 checkpoint
ckpt = torch.load('multi.pt', map_location='cpu')

print("=" * 70)
print("CHECKPOINT 诊断分析")
print("=" * 70)

# 1. checkpoint 的顶层键
print("\n【Checkpoint 顶层键】")
for key in ckpt.keys():
    print(f"  {key}")

# 2. 编码器部分的分析
print("\n【编码器部分分析】")
encoder_ckpt = get_keys(ckpt, 'encoder_firststage')
print(f"  编码器参数总数: {len(encoder_ckpt)}")

# 统计 seq_adapters 相关参数
seq_adapters_keys = [k for k in encoder_ckpt.keys() if 'seq_adapters' in k]
print(f"  seq_adapters 相关参数数: {len(seq_adapters_keys)}")

if seq_adapters_keys:
    print(f"  seq_adapters 样本:")
    for key in seq_adapters_keys[:10]:
        print(f"    - {key}")

# 3. 当前模型的编码器结构
print("\n【当前模型编码器结构】")
opts = Namespace(
    input_nc=3,
    device='cpu'
)
model_encoder = backbone_encoders.EfficientEncoder(50, 'ir_se', opts)
model_encoder_state = model_encoder.state_dict()

print(f"  模型编码器参数总数: {len(model_encoder_state)}")

# 统计 adapter 相关参数
adapter_keys = [k for k in model_encoder_state.keys() if 'adapter' in k]
print(f"  adapter 相关参数数: {len(adapter_keys)}")

if adapter_keys:
    print(f"  adapter 样本:")
    for key in adapter_keys[:10]:
        print(f"    - {key}")

# 4. 对比分析
print("\n【对比分析】")
model_keys = set(model_encoder_state.keys())
ckpt_keys = set(encoder_ckpt.keys())

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"  缺失的参数数: {len(missing)}")
if missing and len(missing) <= 10:
    for key in missing:
        print(f"    - {key}")
elif missing:
    for key in list(missing)[:5]:
        print(f"    - {key}")
    print(f"    ... 还有 {len(missing) - 5} 个")

print(f"\n  多余的参数数: {len(unexpected)}")
if unexpected and len(unexpected) <= 10:
    for key in unexpected:
        print(f"    + {key}")
elif unexpected:
    for key in list(unexpected)[:5]:
        print(f"    + {key}")
    print(f"    ... 还有 {len(unexpected) - 5} 个")

# 5. 解码器部分
print("\n【解码器部分分析】")
decoder_ckpt = get_keys(ckpt, 'decoder')
model_decoder = backbone_encoders.Generator(1024, 512, 8) if hasattr(backbone_encoders, 'Generator') else None

if model_decoder:
    model_decoder_state = model_decoder.state_dict()
    print(f"  Checkpoint 解码器参数数: {len(decoder_ckpt)}")
    print(f"  模型解码器参数数: {len(model_decoder_state)}")
else:
    print(f"  Checkpoint 解码器参数数: {len(decoder_ckpt)}")

print("\n" + "=" * 70)