import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from attrdict import AttrDict
import os
from os import path as osp

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))

class Encoder(nn.Module):
	def __init__(
		self,
		input_channels=45,
		input_height=16,
		input_width=16,
		conv_channels=[64, 32,32],
		kernel_sizes=[3, 3,3],
		strides=[1, 1,1],
		paddings=[1, 1,1],
		pooling='max',
		pool_kernels=[2, 2,2],
		fc_hidden_dims=[32,16,8],
		n_H=4,
		activation=nn.ReLU()
	):
		super(Encoder, self).__init__()

		self.activation = activation
		self.conv_layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		in_channels = input_channels
		H_in, W_in = input_height, input_width

		# Define convolutional and pooling layers
		for i, (out_channels, kernel_size, stride, padding) in enumerate(
			zip(conv_channels, kernel_sizes, strides, paddings)
		):
			# Convolutional layer
			conv = nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding
			)
			self.conv_layers.append(conv)

			# Update spatial dimensions after convolution
			H_in = self._compute_output_dim(H_in, kernel_size, stride, padding)
			W_in = self._compute_output_dim(W_in, kernel_size, stride, padding)

			# Pooling layer
			if pooling == 'max':
				pool = nn.MaxPool2d(kernel_size=pool_kernels[i])
			elif pooling == 'avg':
				pool = nn.AvgPool2d(kernel_size=pool_kernels[i])
			else:
				pool = None
			self.pool_layers.append(pool)

			# Update spatial dimensions after pooling
			if pool is not None:
				H_in = self._compute_output_dim(H_in, pool_kernels[i], pool_kernels[i], 0)
				W_in = self._compute_output_dim(W_in, pool_kernels[i], pool_kernels[i], 0)

			in_channels = out_channels

		# Calculate the flattened feature dimension after convolutions
		self.feature_dim = H_in * W_in * in_channels

		# Define fully connected layers
		fc_dims = [self.feature_dim] + list(fc_hidden_dims) + [n_H]
		self.fc_layers = nn.ModuleList()
		self.fc_batch_norms = nn.ModuleList()
		for i in range(len(fc_dims) - 1):
			in_dim = fc_dims[i]
			out_dim = fc_dims[i + 1]
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
			if i < len(fc_dims) - 2:
				# Add BatchNorm1d layer for all but the last FC layer
				self.fc_batch_norms.append(nn.BatchNorm1d(out_dim))

	def _compute_output_dim(self, size, kernel_size, stride, padding):
		return (size + 2 * padding - kernel_size) // stride + 1

	def forward(self, x):
		# Permute input to (B, C, H, W)
		x = x.permute(0, 3, 1, 2)

		# Apply convolutional and pooling layers
		for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
			x = self.activation(conv_layer(x))
			if pool_layer is not None:
				x = pool_layer(x)

		# Flatten the output from convolutional layers
		x = x.reshape(x.size(0), -1)

		# Apply fully connected layers with batch normalization
		for i, fc_layer in enumerate(self.fc_layers[:-1]):
			x = fc_layer(x)
			x = self.activation(x)
			x = self.fc_batch_norms[i](x)

		# Output layer without batch normalization and activation
		x = self.fc_layers[-1](x)

		return x


class Decoder(nn.Module):
	def __init__(
		self,
		n_H=4,
		fc_hidden_dims=[8,16,32],
		output_dim=45,
		activation=nn.ReLU()
	):
		super(Decoder, self).__init__()

		self.activation = activation
		self.output_dim = output_dim

		# Define fully connected layers
		fc_dims = [n_H] + list(fc_hidden_dims) + [self.output_dim]
		self.fc_layers = nn.ModuleList()
		self.fc_batch_norms = nn.ModuleList()
		for i in range(len(fc_dims) - 1):
			in_dim = fc_dims[i]
			out_dim = fc_dims[i + 1]
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
			if i < len(fc_dims) - 2:
				# Add BatchNorm1d layer for all but the last FC layer
				self.fc_batch_norms.append(nn.BatchNorm1d(out_dim))

	def forward(self, x):
		# x is of shape (B, n_H)
		# Apply fully connected layers with batch normalization
		for i, fc_layer in enumerate(self.fc_layers[:-1]):
			x = fc_layer(x)
			x = self.activation(x)
			x = self.fc_batch_norms[i](x)
			

		# Output layer without batch normalization and activation
		x = self.fc_layers[-1](x)

		# Apply softmax over the output dimension
		# x = F.log_softmax(x, dim=-1)

		return x


class fuel_autoencoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(fuel_autoencoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

		self.LANDFIRE_METADATA_PATH = osp.join(SCRIPT_DIR,"landfire_metadata.csv")

		lf_meta = pd.read_csv(self.LANDFIRE_METADATA_PATH)
		lf_meta.drop(index = 0,inplace= True)
		self.landfire_fuel_classes = dict(zip(lf_meta['VALUE'],lf_meta['FBFM40']))
		self.landfire_class_values = list(self.landfire_fuel_classes.keys())

		self.fuel_classes_ordered = {i:self.landfire_fuel_classes[x] for i,x in enumerate(self.landfire_class_values)}
		self.fuel_class_map = {x:i for i,x in enumerate(self.landfire_class_values)}

		self.class_viz_rgb = {row['FBFM40']:(row['R'],row['G'],row['B']) for i,row in lf_meta.iterrows()}

		self.encode_mode = True

	def replace_categories(self,value):
		try:
			return self.fuel_class_map[value]
		except KeyError:
			return float('nan')

	def get_pcts(self,arr): # takes (B,H,W) array
		B,H,W = arr.shape
		n_cats = len(self.landfire_class_values)
		res = torch.zeros((B,n_cats))
		for i,img in enumerate(arr):
			for k,cat in enumerate(kys):
				res[i,k] = torch.sum(arr[i] == cat).item()/(H*W)
		return res

	def process_data_batch(self,data_batch,get_labels = False,onehot = True,target_shape = (16,16)):
		data = data_batch['fbfm'].clone()
		assert (data.shape[1] == target_shape[0]*target_shape[1]),f"Data shape is {data.shape} while target shape is {target_shape}."

		data = data.reshape((data.shape[0],target_shape[0],target_shape[1]))
		if get_labels:
			labels = self.get_pcts(data)
		data.apply_(self.replace_categories)
		if onehot:
			data = F.one_hot(data.long(),num_classes = len(self.landfire_class_values)).float()
		if get_labels:
			return data,labels
		else:
			return data


	def forward(self, x):
		
		# x = self.process_data_batch(x,get_labels = False,onehot = True,target_shape = (16,16))

		latent = self.encoder(x)
		if self.encode_mode:
			return latent
		else:
			output_vector = self.decoder(latent)
			return output_vector

	def encode(self,x):
		x = self.process_data_batch(x,get_labels = False,onehot = True,target_shape = (16,16))

		with torch.no_grad():
			latent = self.encoder(x)
		return latent
			

def load(
	weights_file = 'fuel_autoencoder_weights.pth',
	config_file = 'fuel_autoencoder_config.json',
	):
	
	with open(osp.join(SCRIPT_DIR, config_file),'r') as json_file:
		config = AttrDict(json.load(json_file))

	encoder = Encoder(
		input_channels=45,
		input_height=16,
		input_width=16,
		conv_channels=config.conv_channels,
		kernel_sizes=config.kernel_sizes,
		strides=config.strides,
		paddings=config.paddings,
		pooling=config.pooling,
		pool_kernels=config.pool_kernels,
		fc_hidden_dims=config.encoder_hidden_dims,
		n_H=config.latent_dim,)

	decoder = Decoder(
		n_H=config.latent_dim,
		fc_hidden_dims=config.decoder_hidden_dims,
	)

	model = fuel_autoencoder(encoder,decoder)
	
	model_wts = torch.load(osp.join(SCRIPT_DIR, weights_file),weights_only = True)
	model.load_state_dict(model_wts)
	
	return model, "fuel_autoencoder"

