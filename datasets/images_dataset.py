from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np

class ImagesDataset(Dataset):

	def __init__(self, source_root, train, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		if train:
			self.source_paths =self.source_paths[:int(len(self.source_paths)*0.9)]
		else:
			self.source_paths = self.source_paths[int(len(self.source_paths)*0.9):]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		path = self.source_paths[index]
		img = Image.open(path)
		img = img.convert('RGB') if self.opts.label_nc == 0 else img.convert('L')
  
		img = np.array(img)
  
		from_imgs = list()
		for i in range(1,8):
			from_imgs.append(Image.fromarray(img[:,i*256:(i+1)*256,:]))
   
		to_img = Image.fromarray(img[:,:256,:])
  
		if np.random.uniform(0, 1) < 0.5:
			from_imgs = [from_img.transpose(Image.FLIP_LEFT_RIGHT) for from_img in from_imgs] 
			to_img = to_img.transpose(Image.FLIP_LEFT_RIGHT)

		if self.target_transform:
			to_img = self.target_transform(to_img)

		if self.source_transform:
			from_imgs = [self.source_transform(from_img) for from_img in from_imgs]
   
		idx = np.random.randint(7)
		from_img = from_imgs[idx]

		return from_img, to_img
