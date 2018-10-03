"""
Written by Matteo Dunnhofer - 2018

Data transformer class
"""
from PIL import Image
import numpy as np
import scipy.misc as misc
#from skimage.color import rgb2gray
#from skimage.transform import resize

class DataTransformer(object):
	"""
	Defines an object that is responsible to transform the
	example images

	img: numpy array
	"""
	
	def __init__(self, cfg):
		self.cfg = cfg

	def preprocess(self, img, crop):
		"""
		Args:
			img: a image matrix

		Returns:
			a processed image matrix
		"""
		
		#img_g = Image.fromarray(img).convert('L')
		#img_r = img_g.resize((self.cfg.IMG_SIZE[0], self.cfg.IMG_SIZE[1]), Image.ANTIALIAS)
		#img = np.array(img_r)
		#img = np.reshape(img, (self.cfg.IMG_SIZE[0], self.cfg.IMG_SIZE[1], 1))

		#img_g = Image.fromarray(img_g)
		#img_r.save('im.png')

		#img = np.dot(img[:,:,:3], [0.299, 0.587, 0.114])
		#img = misc.imresize(img, [self.cfg.IMG_SIZE[0], self.cfg.IMG_SIZE[1]], 'bilinear')
		#img = img.astype(np.float32) / 128.0 - 1.0

		img = img[crop[0]:crop[1]+160, :160]
		img = img.mean(2)
		img = img.astype(np.float32)
		img *= (1.0 / 255.0)
		img = misc.imresize(img, [80, crop[2]], 'bilinear')
		img = misc.imresize(img, [self.cfg.OBSERVATION_SIZE[0], self.cfg.OBSERVATION_SIZE[1]], 'bilinear')
		img = np.reshape(img, (self.cfg.OBSERVATION_SIZE[0], self.cfg.OBSERVATION_SIZE[1], 1))

		return img
