import torch
from torch import nn
from models.encoders import backbone_encoders
from models.stylegan2.model import Generator

"""
e.g. 
d: multi.pt
k = "encoder_firststage.input_layer.0.weight"
name = "encoder_firststage"
len(name) = 20
k[len(name) + 1:] = k[21:] = encoder_firststage[input_layer.0.weight] = v
从特定模块(name)提取这个模块的权重字典(d_filt)
"""
def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']# 抽取真实权重字典
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt

class FaceFrontalizier(nn.Module):
	def __init__(self, opts):# opts包含模型的各种配置参数
		super(FaceFrontalizier, self).__init__()
		self.set_opts(opts)
		self.encoder = backbone_encoders.EfficientEncoder(50, 'ir_se', self.opts)

		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()
		# 冻结编码器和解码器的参数
		self.freeze_encoder()
		self.freeze_decoder()
  
# freeze即冻结模型参数，防止在训练过程中被更新，通常用于微调预训练模型 
	def freeze_decoder(self):
		print('freezing decoder ...')
		for param in self.decoder.parameters():
			param.requires_grad = False
   
	def freeze_encoder(self):
		print('freezing encoder ...')
		for name, param in self.encoder.named_parameters():
			if 'adapter_layer' not in name:
				param.requires_grad = False

# 从checkpoint_path加载预训练权重
	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):
			print('Loading face frontalization model from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)

			#ckpt是一个包含模型权重的字典 ckpt即checkpoint
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')

			#将ckpt中权重分别导入到编码器和解码器
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=False)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)

			# 从checkpoint加载latent_avg（latent空间的均值向量）
			self.__load_latent_avg(ckpt)

			# 如果在训练的话则从之前的检查点加载编码器和解码器，进而继续训练
		elif (self.opts.checkpoint_path is not None) and self.opts.is_training:
			print('Loading E2Style from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)		

# forward：使用stylegan2的G作为decoder，将输入的code/解码后的latent映射回图像空间，是生成的过程
	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else: 
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
									   input_is_latent=input_is_latent,
									   randomize_noise=randomize_noise,
									   return_latents=return_latents)

		if resize: 
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None): 
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
