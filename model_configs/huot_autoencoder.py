import torch
import torch.nn as nn

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, use_maxpool=False):
		super(ResBlock, self).__init__()
		
		
		# Main path
		self.leaky_relu1 = nn.LeakyReLU()
		self.dropout1 = nn.Dropout(0.1)
		# interior option
		if use_maxpool:
			self.interior_layer = nn.MaxPool2d(2)
			# skip connection
			self.dropout_skip = nn.Dropout(0.1)
			self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,padding = 1)
		else:
			self.interior_layer = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same')
			# skip connection
			self.dropout_skip = nn.Dropout(0.1)
			self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
			
		self.leaky_relu2 = nn.LeakyReLU()
		self.dropout2 = nn.Dropout(0.1)
		self.conv_interior = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
		self.dropout3 = nn.Dropout(0.1)   
		
	def forward(self, x):
		skip = x
		
		# Main path
		out = self.leaky_relu1(x)
		out = self.dropout1(out)
		out = self.interior_layer(out)
		
		out = self.leaky_relu2(out)
		out = self.dropout2(out)
		out = self.conv_interior(out)
		out = self.dropout3(out)
		
		skip = self.conv_skip(skip)
		skip = self.dropout_skip(skip)	
		
		return out + skip
		
class FireAutoencoder(nn.Module):
	def __init__(self, input_channels):
		super(FireAutoencoder, self).__init__()
		
		# initial block
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding='same')
		self.leaky_relu = nn.LeakyReLU()
		self.dropout1 = nn.Dropout(0.1)
		self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
		
		self.conv_skip1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding='same')
		self.dropout_skip1 = nn.Dropout(0.1)
		
		self.resblock1 = ResBlock(16,32,use_maxpool = True)
		self.resblock2 = ResBlock(32,32,use_maxpool = True)
		
		self.upsample1 = nn.Upsample(scale_factor = 2)
		self.resblock3 = ResBlock(32,32,use_maxpool = False)
		self.upsample2 = nn.Upsample(scale_factor = 2)
		self.resblock4 = ResBlock(32,16,use_maxpool = False)
		
		self.final_conv = nn.Conv2d(16,1,kernel_size=3, stride=1, padding='same')
		
	def forward(self, x):
		# initial block
		skip = x 
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.dropout1(x)
		x = self.conv2(x)
		
		skip = self.dropout_skip1(self.conv_skip1(skip))
		
		x = x + skip
		# print(x.shape)
		
		x = self.resblock1(x)
		x = self.resblock2(x)
		x = self.upsample1(x)
		x = self.resblock3(x)
		x = self.upsample2(x)
		x = self.resblock4(x)
		
		x = self.final_conv(x)
		
		return x

def init_weights(m):

	if isinstance(m, nn.Conv2d):
		# Initialize the weights using Kaiming normal initialization
		torch.nn.init.kaiming_normal_(m.weight)
		
		# If the module has a bias term, initialize it to zeros
		if m.bias is not None:
			torch.nn.init.zeros_(m.bias)


def load(config):

	torch.manual_seed(42)

	model = FireAutoencoder(input_channels = config['in_channels'])

	model.apply(init_weights)

	return model


