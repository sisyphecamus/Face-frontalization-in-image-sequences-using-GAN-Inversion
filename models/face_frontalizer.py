import torch
from torch import nn
from models.encoders import backbone_encoders
from models.stylegan2.model import Generator

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt


class FaceFrontalizier(nn.Module):

	def __init__(self, opts):
		super(FaceFrontalizier, self).__init__()
		self.set_opts(opts)
		self.encoder = backbone_encoders.EfficientEncoder(50, 'ir_se', self.opts)

		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()
		self.freeze_encoder()
		self.freeze_decoder()
  
  
	def freeze_decoder(self):
		print('freezing decoder ...')
		for param in self.decoder.parameters():
			param.requires_grad = False
   
	def freeze_encoder(self):
		print('freezing encoder ...')
		for name, param in self.encoder.named_parameters():
			if 'adapter_layer' not in name:
				param.requires_grad = False


	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):
			print('Loading face frontalization model from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is not None) and self.opts.is_training:
			print('Loading E2Style from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=False)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)		


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
		images, result_latent = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)

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
