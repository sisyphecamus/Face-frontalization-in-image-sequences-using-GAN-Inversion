import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
# 使用helpers中的辅助函数
from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
"""
预先了解：ResNet
ResNet（Residual Network）是一种深度卷积神经网络架构，旨在解决深层网络训练中的退化问题。
它通过引入“残差连接”（skip connections）允许梯度直接流过网络，从而使得非常深的网络也能有效训练。
ResNet的核心思想是让网络学习残差映射，而不是直接学习期望的输出映射。
ResNet的基本构建块是“残差块（Bottleneck Residual Block）”，它包含两个或更多的卷积层，并通过一个快捷连接将输入直接添加到输出。
这种结构使得网络能够更容易地学习恒等映射，从而缓解了深层网络中的梯度
"""

# 数据预处理

# 把图像特征映射成对 latent 的微调项，然后把它加回原始 latent 子向量 w 上
# 允许网络“利用空间上下文”调整每一层的 latent（相当于条件化 latent）
# 提高编码器对不同角度/部位细节的表达能力（AdapterBlock在编码器的不同层级上都有）
# 多帧/多视角场景下，adapter 可以把帧级别的视觉信息映射到对应的 StyleGAN 层，改善最终生成质量
class AdapterBlock(Module):
    def __init__(self, in_channel, num_module):
        super().__init__()
        self.num_module = num_module
        self.adapter = Sequential(BatchNorm2d(in_channel), # 归一化
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),# 固定采样7*7
                                         Flatten(),# 拉直为49
                                         Linear(in_channel * 7 * 7, 2 * in_channel),# 降维，特征投影
                                         nn.GELU(),# 激活函数
                                         Linear(2 * in_channel, 512 * num_module))# 每个module都有512维
        

    def forward(self, x , w):
        out = self.adapter(x).view(-1, self.num_module, 512) 
        return w + out

class EfficientEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None): # ir = improved residual
        super(EfficientEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se' # se = squeeze-excitation
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR 
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        
        self.adapter_layer_3 = AdapterBlock(256, 9)
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        
        self.adapter_layer_4 = AdapterBlock(128, 5)
        
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        
        self.adapter_layer_5 = AdapterBlock(64, 4)
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        lc_part_4 = self.adapter_layer_5(x, lc_part_4)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        lc_part_3 = self.adapter_layer_4(x, lc_part_3)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)
        lc_part_2 = self.adapter_layer_3(x, lc_part_2)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x