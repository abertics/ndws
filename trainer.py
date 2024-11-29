# trainer object for neural net testing

import warnings
import torch
import math
import sys
from tqdm import trange
from tqdm.auto import tqdm
import numpy as np
import os
import json
import h5py
import matplotlib.pyplot as plt

import sys
import re
import importlib
import importlib.util
import shutil
import deepdish as dd

from os import path as osp

import torch.nn as nn 
import torch.nn.functional as F

from sklearn.metrics import precision_recall_curve,auc
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser


def aggregate(arr_dict):
    res = []
    for key in arr_dict:
        res.append(arr_dict[key])
    return np.concatenate(res,axis = 0).flatten()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def _import_module(module_path_or_name):
    """Dynamically imports a module from a filepath or a module name."""
    module, name = None, None

    if module_path_or_name.endswith('.py'):

        if not os.path.exists(module_path_or_name):
            raise RuntimeError('File {} does not exist.'.format(module_path_or_name))

        file_name = module_path_or_name
        module_name = os.path.basename(os.path.splitext(module_path_or_name)[0])

        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            # Use importlib to load the module from the file
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            if spec is None:
                raise ImportError(f"Cannot load specification for '{module_name}' from '{file_name}'")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    else:
        module = importlib.import_module(module_path_or_name)
        module_name = module.__name__

    if module:
        name = module_name.split('.')[-1].split('/')[-1]

    return module, name

def load(conf_path, *args, **kwargs):
    """Loads a config."""

    module, name = _import_module(conf_path)
    try:
        load_func = module.load
    except AttributeError:
        raise ValueError("The config file should specify 'load' function but no such function was "
                           "found in {}".format(module.__file__))

    print("Loading '{}' from {}".format(module.__name__, module.__file__))
    return load_func(*args, **kwargs)

def initialize_experiment(name,exp_dir = 'model_experiments'):
	if not osp.exists(exp_dir):
		os.makedirs(exp_dir)
	if not osp.exists(osp.join(exp_dir,name)):
		os.makedirs(osp.join(exp_dir,name))
	return osp.join(exp_dir,name)


def create_abbr_filename(config):
	# Dictionary mapping config keys to their abbreviations
	abbreviations = {
		'batch_size': 'bs',
		'learning_rate': 'lr',
		'features_to_drop':'drop',
		'in_channels':'inCh',
		'early_stop':'es',
		'early_stop_start':'esS',
		'patience':'pat',
		'focal_gamma':'gamma',
		'num_attention_blocks':'nAtt',
		'dropout_prob':'drp',
		# 'final_channels':'fCh',
		'lambda_sparse':'lSp',
	}

	run_name = ''
	for key, value in config.items():
		if isinstance(value, (int, float, str, bool)) and key in abbreviations:
			run_name += f"{abbreviations[key]}{value}_"
	
	return run_name

def create_next_folder(experiment_dir,basename = ''):
	i = 1
	base_path = osp.join(experiment_dir,basename)
	while True:
		# Construct folder name
		folder_name = f"{base_path}{i}"
		# Check if folder exists
		if not osp.exists(folder_name):
			# Create the folder since it does not exist
			os.makedirs(folder_name)
			return folder_name
		i += 1  # Increment to try the next number if this folder already exists
		

class WeightedBCELoss(nn.Module):
    
    def __init__(self, weights=torch.tensor([1,1]), device='cpu', gamma=0.0, lambda_sparse=0.0):
        super(WeightedBCELoss, self).__init__()
        
        self.weights = weights.to(device)
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.device = device
        
        if gamma == 0:
            self.logLoss = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, preds, labels):
        # For logits, we need to concatenate the negative logit with the positive logit
        if preds.shape[1] == 1:
            # Convert single logit to two logits (negative and positive)
            preds = torch.cat([-preds, preds], dim=1)
            
        labels = labels.long()
        
        # Compute main loss (focal or regular BCE)
        if self.gamma == 0:
            main_loss = self.logLoss(preds, labels)
        else:
            # Compute focal loss when gamma > 0
            probs = F.softmax(preds, dim=1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            
            # Clamp pt to prevent log(0) and ensure numerical stability
            pt = pt.clamp(min=1e-6, max=1.0 - 1e-6)
            
            # Get weights for each sample
            weight = self.weights[labels]
            
            # Compute focal weight
            focal_weight = (1 - pt).pow(self.gamma)
            
            # Compute focal loss
            focal_loss = -weight * focal_weight * torch.log(pt)
            main_loss = focal_loss.mean()
        
        # Add L1 sparsity term if lambda_sparse > 0
        if self.lambda_sparse > 0:
            # Get the positive logits (second channel)
            positive_logits = preds[:, 1]
            # Compute L1 sparsity on the positive logits
            sparsity_loss = torch.mean(torch.abs(positive_logits))
            return main_loss + self.lambda_sparse * sparsity_loss
            
        return main_loss
		

class Trainer:
	"""
	Trainer class that eases the training of a PyTorch model.
	Parameters
	----------
	model : torch.nn.Module
		The model to train.
	criterion : torch.nn.Module
		Loss function criterion.
	optimizer : torch.optim
		Optimizer to perform the parameters update.
	epochs : int
		The total number of iterations of all the training 
		data in one cycle for training the model.
	scaler : torch.cuda.amp
		The parameter can be used to normalize PyTorch Tensors 
		using native functions more detail:
		https://pytorch.org/docs/stable/index.html.
	lr_scheduler : torch.optim.lr_scheduler
		A predefined framework that adjusts the learning rate 
		between epochs or iterations as the training progresses.
	Attributes
	----------
	train_losses_ : torch.tensor
		It is a log of train losses for each epoch step.
	val_losses_ : torch.tensor
		It is a log of validation losses for each epoch step.
	"""
	def __init__(
		self, 
		model, 
		criterion, 
		optimizer,
		epochs,
		lr_scheduler=None, 
		device=None,
		print_grads = False,
		early_stop=True,
		patience=5,
		early_stop_start=20,
		min_delta=0,
		weights_file = None,
	):
		self.criterion = criterion
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.device = self._get_device(device)
		self.epochs = epochs
		self.model = model.to(self.device)
		self.print_grads = print_grads
		self.early_stop = early_stop
		self.patience = patience
		self.min_delta = min_delta
		self.best_val_loss = float('inf')
		self.counter = 0
		self.early_stop_start = early_stop_start
		self.weights_file = weights_file


	def _print_average_gradients(self):
		avg_gradients = []
		for name, parameter in self.model.named_parameters():
			if parameter.requires_grad and parameter.grad is not None:
				grad_norm = parameter.grad.norm().item()
				avg_gradients.append(grad_norm)
				print(f"Average gradient for {name}: {grad_norm}")
		print(f"Overall average gradient norm: {np.mean(avg_gradients)}")
		
	def fit(self, train_loader, val_loader):
		"""
		Fit the model using the given loaders for the given number
		of epochs.
		
		Parameters
		----------
		train_loader : 
		val_loader : 
		"""
		# attributes  
		self.train_losses_ = torch.zeros(self.epochs)
		self.val_losses_ = torch.zeros(self.epochs)
		self.stop_epoch = None
		self.best_loss = float('inf')

		# Initialize plotly figure
		self.fig = make_subplots(rows=1, cols=2,
								subplot_titles=('Training Loss', 'Validation Loss'))
		
		# Set initial layout
		self.fig.update_layout(
			height=400, width=1000,
			showlegend=True,
			title_text="Training Progress",
			xaxis_title="Batch",
			xaxis2_title="Batch",
			yaxis_title="Loss",
			yaxis2_title="Loss"
		)
		
		self.train_losses = []  # Store all training losses
		self.val_losses = []    # Store all validation losses
		
		# Save the initial plot to an HTML file with auto-refresh and open it in a browser
		html_path = 'training_progress.html'
		html_content = self.fig.to_html(
			include_plotlyjs=True,
			full_html=True,
			default_height='100vh'
		)
		
		# Insert auto-refresh meta tag into the HTML head
		html_content = html_content.replace(
			'<head>',
			'<head><meta http-equiv="refresh" content="2">'  # Refresh every 2 seconds
		)
		
		with open(html_path, 'w') as f:
			f.write(html_content)
		
		webbrowser.open('file://' + os.path.realpath(html_path))
		
		# ---- train process ----
		for epoch in trange(1, self.epochs + 1, desc='Training Model on {} epochs'.format(self.epochs)):
			if self.stop_epoch is None:
				self._train_one_epoch(train_loader, epoch)
				val_loss = self._evaluate(val_loader, epoch)
				print(f'Validation loss: {val_loss:.4f}')
				
				if val_loss < self.best_loss:
					self.best_loss = val_loss
					torch.save(self.model.state_dict(), self.weights_file)
					
				# Early stopping logic
				if self.early_stop and epoch > self.early_stop_start:
					if val_loss < self.best_val_loss - self.min_delta:
						self.best_val_loss = val_loss
						self.counter = 0
					else:
						self.counter += 1
						if self.counter >= self.patience:
							self.stop_epoch = epoch
							print(f"Early stopping at epoch {self.stop_epoch}")
			else:
				self.train_losses_[epoch - 1] = self.train_losses_[epoch - 2]
				self.val_losses_[epoch - 1] = self.val_losses_[epoch - 2]
		
		# Save losses to h5 file after training
		loss_data = {
			'train_losses': self.train_losses_,
			'val_losses': self.val_losses_,
		}
		dd.io.save('training_losses.h5', loss_data)

	def _update_plots(self):
		"""Helper method to update the plotly figures"""
		with self.fig.batch_update():
			# Clear the previous traces
			self.fig.data = []
			
			# Update training loss plot
			self.fig.add_trace(
				go.Scatter(x=list(range(len(self.train_losses))), 
						  y=self.train_losses, 
						  name='Training Loss', 
						  line=dict(color='blue')),
				row=1, col=1
			)
			
			# Update validation loss plot
			self.fig.add_trace(
				go.Scatter(x=list(range(len(self.val_losses))), 
						  y=self.val_losses, 
						  name='Validation Loss', 
						  line=dict(color='red')),
				row=1, col=2
			)
			
			# Update layout to use log scale for both y-axes
			self.fig.update_layout(
				title_text=f"Training Progress - Batch {len(self.train_losses)}",
				yaxis_type="log",  # Set training loss y-axis to log scale
				yaxis2_type="log"  # Set validation loss y-axis to log scale
			)
			
			# Save updated plot with auto-refresh
			html_content = self.fig.to_html(
				include_plotlyjs=True,
				full_html=True,
				default_height='100vh'
			)
			html_content = html_content.replace(
				'<head>',
				'<head><meta http-equiv="refresh" content="10">'
			)
			with open('training_progress.html', 'w') as f:
				f.write(html_content)

	def _train_one_epoch(self, data_loader, epoch):
		self.model.train()
		losses = []
		
		with tqdm(data_loader, unit=" training-batch", colour="green") as training:
			try:
				for batch_idx, (data, labels) in enumerate(training):
					data, labels = data.to(self.device), labels.to(self.device)
					
					# forward pass
					self.optimizer.zero_grad() # remove gradient from previous passes
					
					preds = self.model(data)
					loss = self.criterion(preds.float(), labels)
					
					if not math.isfinite(loss):
						msg = f"Loss is {loss}, stopping training!"
						warnings.warn(msg)
						sys.exit(1)
					
					loss.backward()

					# parameters update
					self.optimizer.step()
					if self.lr_scheduler is not None:
						self.lr_scheduler.step()
					
					loss_val = loss.item()
					losses.append(loss_val)
					self.train_losses.append(loss_val)
					
					# Update plots after each batch
					self._update_plots()
					
					# Update progress bar
					training.set_postfix(loss=loss_val)
					
			except IndexError:
				print('IndexError happened')
		
		# Check and print the average gradients here
		if self.print_grads:
			self._print_average_gradients()
		self.train_losses_[epoch - 1] = np.mean(losses)
		print(f'Train loss: {self.train_losses_[epoch - 1]:.4f}')

	
	@torch.inference_mode()
	def _evaluate(self, data_loader, epoch):
		self.model.eval()
		losses = []
		with torch.no_grad():
			with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
				try:
					for data, labels in evaluation:
						evaluation.set_description(f"Validation")
						data, labels = data.to(self.device), labels.to(self.device)
						preds = self.model(data)
						loss = self.criterion(preds.float(), labels)
						loss_val = loss.item()
						losses.append(loss_val)
						self.val_losses.append(loss_val)
						
						# Update plots after each validation batch
						self._update_plots()
						
						evaluation.set_postfix(loss=loss_val)
						
				except IndexError:
					print('IndexError happened')

				self.val_losses_[epoch - 1] = np.mean(losses)
		return np.mean(losses)

	def _get_device(self, _device):
		if _device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			msg = f"Device was automatically selected: {device}"
			warnings.warn(msg)
			return device
		return _device
	

class Experiment:

	def __init__(self, name, config):
		self.name = name
		self.config = config
		self.experiment_dir = initialize_experiment(name)
		self.device = config['device']

		self.model_file = config['model_config_file']
		self.data_file = config['data_config_file']

		self.loaders = load(self.data_file, self.config) # load dataloaders

		self.features = self.loaders['train'].features
		self.config['features'] = self.features

		self.in_channels = len(self.features)
		self.config['in_channels'] = self.in_channels

		self.model = load(self.model_file, self.config) # load model

		self.run_dir = osp.join(self.experiment_dir, create_abbr_filename(self.config))

		self.model_dir = create_next_folder(self.run_dir)
		print(self.model_dir)

		with open(osp.join(self.model_dir, 'config.json'), 'w') as f:
			json.dump(self.config, f) # save config

		# Save copies of model and data configuration files
		shutil.copy2(self.model_file, osp.join(self.model_dir, osp.basename(self.model_file)))
		shutil.copy2(self.data_file, osp.join(self.model_dir, osp.basename(self.data_file)))

		
		self.criterion = WeightedBCELoss(
			device=self.device,
			weights=torch.tensor(self.config['loss_weights']),
			gamma=self.config['focal_gamma'],
			lambda_sparse=self.config['lambda_sparse'],
			)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

		self.weights_file = osp.join(self.model_dir, 'weights.pth')
		self.trainer = Trainer(
			self.model, 
			self.criterion, 
			self.optimizer,
			self.config['epochs'],
			device=self.config['device'],
			print_grads = False,
			early_stop=self.config['early_stop'],
			patience=self.config['patience'],
			early_stop_start=self.config['early_stop_start'],
			min_delta=0,
			weights_file=self.weights_file,
		)
		self.train_losses_ = []
		self.val_losses_ = []

	def train(self):
		self.trainer.fit(self.loaders['train'], self.loaders['eval'])

		self.train_losses_ = self.trainer.train_losses_
		self.val_losses_ = self.trainer.val_losses_

		torch.save(self.model.state_dict(), self.weights_file)

		fig,ax = plt.subplots()
		ax.plot(self.train_losses_, label='Train Loss')
		ax.plot(self.val_losses_, label='Validation Loss')
		ax.legend()
		plt.savefig(osp.join(self.model_dir, 'losses.png'))


class Evaluator:
	def __init__(self, name, config,exp_num = None,dirname = None):
		if dirname is None:
			self.name = name
			self.config = config
			
			
			self.experiment_dir = initialize_experiment(name)

			self.run_dir = osp.join(self.experiment_dir, create_abbr_filename(self.config))

			if not osp.exists(self.run_dir):
				raise ValueError('Experiment directory does not exist')
			if exp_num is None:
				# Get all subdirectories in run_dir that are numbers
				exp_dirs = [d for d in os.listdir(self.run_dir) if d.isdigit()]
				if not exp_dirs:
					raise ValueError('No experiment directories found')
				# Get highest numbered directory
				exp_num = max(map(int, exp_dirs))
				self.model_dir = osp.join(self.run_dir, str(exp_num))
			else:
				self.model_dir = osp.join(self.run_dir, f'exp{exp_num}')

			if not osp.exists(self.model_dir):
				raise ValueError('Model directory does not exist')
		else:
			self.model_dir = dirname

		self.config_file = osp.join(self.model_dir, 'config.json')
		with open(self.config_file, 'r') as f:
			self.config = json.load(f)

		self.model_file = self.config['model_config_file']
		self.data_file = self.config['data_config_file']
		self.device = self.config['device']
		
		self.model_file = osp.join(self.model_dir, osp.basename(self.model_file))
		self.data_file = osp.join(self.model_dir, osp.basename(self.data_file))

		self.model = load(self.model_file, self.config)
		self.loaders = load(self.data_file, self.config)

		self.weights_file = osp.join(self.model_dir, 'weights.pth')
		if not osp.exists(self.weights_file):
			raise ValueError('Weights file does not exist')
		self.model.load_state_dict(torch.load(self.weights_file,weights_only=True))

		print("Model loaded from {}".format(self.model_dir))

		self.pred_file = None
		self.label_file = None
		self.firemask_file = None
		self.best_threshold = None

		self.sample_file = osp.join(self.model_dir,'sample_data.h5')

	def perform_inference(self,regen = False):
		test_loader = self.loaders['test']
		
		self.model.to(self.device)
		self.model.eval()
		
		# Create HDF5 files for predictions and labels
		self.pred_file = osp.join(self.model_dir, 'predictions.h5')
		self.label_file = osp.join(self.model_dir, 'labels.h5')
		self.firemask_file = osp.join(self.model_dir, 'firemasks.h5')

		if not regen and (osp.exists(self.pred_file) or osp.exists(self.label_file)):
			raise ValueError('Predictions or labels file already exists')
			return
		
		
		
		with h5py.File(self.pred_file, 'w') as f_pred, h5py.File(self.label_file, 'w') as f_label, h5py.File(self.firemask_file, 'w') as f_fire:
			# Create groups for batches
			pred_grp = f_pred.create_group('predictions')
			label_grp = f_label.create_group('labels')
			firemask_grp = f_fire.create_group('firemasks')
			print("Beginning inference...")
			
			batch_idx = 0
			with torch.no_grad():
				for data, label in tqdm(test_loader):
					data, label = data.to(self.device), label.to(self.device)
					pred = self.model(data)
					
					# Convert to numpy arrays
					pred_np = pred.cpu().numpy()
					label_np = label.cpu().numpy()
					firemask = data[:,-1,...].cpu().numpy()
					
					# Create datasets for this batch
					pred_grp.create_dataset(
						f'batch_{batch_idx}', 
						data=pred_np,
						chunks=True,
						compression='gzip'
					)
					
					label_grp.create_dataset(
						f'batch_{batch_idx}', 
						data=label_np,
						chunks=True,
						compression='gzip'
					)
					
					firemask_grp.create_dataset(
						f'batch_{batch_idx}', 
						data=firemask,
						chunks=True,
						compression='gzip'
					)
					
					# Store batch size as attribute
					pred_grp[f'batch_{batch_idx}'].attrs['batch_size'] = len(pred_np)
					label_grp[f'batch_{batch_idx}'].attrs['batch_size'] = len(label_np)
					firemask_grp[f'batch_{batch_idx}'].attrs['batch_size'] = len(firemask)
					batch_idx += 1
		
		print(f"Predictions saved to: {self.pred_file}")
		print(f"Labels saved to: {self.label_file}")
		print(f"Firemasks saved to: {self.firemask_file}")
		
		return self.pred_file, self.label_file
	
	def evaluate_pr(self):
		# if osp.exists(osp.join(self.model_dir,'pr_curve.png')):
		# 	print("PR curve already exists, displaying...")
		# 	img = plt.imread(osp.join(self.model_dir,'pr_curve.png'))
		# 	plt.figure(dpi=200)
		# 	plt.imshow(img)
		# 	plt.axis('off')
		# 	plt.show()
		# 	return
		if not osp.exists(self.pred_file) or not osp.exists(self.label_file):
			raise ValueError('Predictions or labels file does not exist')
			
		preds = dd.io.load(self.pred_file)['predictions']
		labels = dd.io.load(self.label_file)['labels']
		persistence = dd.io.load(self.firemask_file)['firemasks']

		preds = sigmoid(aggregate(preds))
		labels = aggregate(labels)
		persistence = aggregate(persistence)

		precision,recall,thresholds = precision_recall_curve(labels,preds)
		auc_score = auc(recall,precision)

		# Compute persistence precision and recall directly since it's a binary mask
		persistence_precision,persistence_recall,persistence_thresholds = precision_recall_curve(labels,persistence)
		persistence_auc = auc(persistence_recall,persistence_precision)

		persistence_f1 = 2 * (persistence_precision[1] * persistence_recall[1]) / (persistence_precision[1] + persistence_recall[1]+1e-6)

		f1_scores = 2 * (precision * recall) / (precision + recall+1e-6)

		# Find the maximum F1 score and the corresponding threshold
		best_f1_index = np.argmax(f1_scores)
		self.best_threshold = thresholds[best_f1_index]
		best_f1 = f1_scores[best_f1_index]

		threshold_precision = precision[best_f1_index]
		threshold_recall = recall[best_f1_index]

		fig,ax = plt.subplots(dpi = 150)
		ax.plot(recall,precision,label=f'AUC-PR = {auc_score:.3f}')
		ax.scatter(persistence_recall[1],persistence_precision[1],label=f'baseline AUC-PR = {persistence_auc:.3f}',color = 'red')
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.text(0.6,0.6,f'Precision: {threshold_precision:.3f}\nRecall: {threshold_recall:.3f}\nF1: {best_f1:.3f}',transform=ax.transAxes)
		ax.text(0.6,0.4,f'Baseline precision: {persistence_precision[1]:.3f}\nBaseline recall: {persistence_recall[1]:.3f}\n Baseline F1: {persistence_f1:.3f}',transform=ax.transAxes)
		ax.legend()
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.title('Precision-Recall Curve')
		plt.savefig(osp.join(self.model_dir,'pr_curve.png'))
		plt.show()

	def evaluate_pr_newOnly(self):
		"""Evaluates precision-recall metrics for only new fire predictions (where persistence=0)"""
		if not osp.exists(self.pred_file) or not osp.exists(self.label_file):
			raise ValueError('Predictions or labels file does not exist')
			
		# Load predictions, labels and persistence masks
		preds = dd.io.load(self.pred_file)['predictions']
		labels = dd.io.load(self.label_file)['labels']
		persistence = dd.io.load(self.firemask_file)['firemasks']

		# Apply sigmoid and aggregate
		preds = sigmoid(aggregate(preds))
		labels = aggregate(labels)
		persistence = aggregate(persistence)

		# Only evaluate on pixels where there was no fire in previous timestep
		new_mask = (persistence == 0)
		new_persistence = persistence[new_mask]
		new_preds = preds[new_mask]
		new_labels = labels[new_mask]

		# Calculate precision-recall metrics
		precision, recall, thresholds = precision_recall_curve(new_labels, new_preds)
		auc_score = auc(recall, precision)

		# Calculate F1 scores and find optimal threshold
		f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
		best_f1_index = np.argmax(f1_scores)
		self.best_threshold = thresholds[best_f1_index]
		best_f1 = f1_scores[best_f1_index]

		threshold_precision = precision[best_f1_index]
		threshold_recall = recall[best_f1_index]

		# Plot results
		fig, ax = plt.subplots(dpi=150)
		ax.plot(recall, precision, label=f'AUC-PR = {auc_score:.3f}')
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.text(0.6, 0.6, f'Precision: {threshold_precision:.3f}\nRecall: {threshold_recall:.3f}\nF1: {best_f1:.3f}', transform=ax.transAxes)
		ax.legend()
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		plt.title('Precision-Recall Curve (New Fires Only)')
		plt.savefig(osp.join(self.model_dir, 'pr_curve_newOnly.png'))
		plt.show()

	def generate_sample(self,batch_num = 0):
		
		test_loader = self.loaders['test']

		for i, (data, label) in enumerate(test_loader):
			if i == batch_num:
				break
		data = data.to(self.device)
		self.model.eval()
		with torch.no_grad():
			pred = sigmoid(self.model(data).cpu().detach())

		sample_pred = pred# [sample_idx]
		sample_label = label# [sample_idx]

		sample_data = {'data':data.cpu(),'label':sample_label,'pred':sample_pred}
		
		dd.io.save(self.sample_file,sample_data)

		print(f"Sample data saved to: {self.sample_file}")

	def plot_sample(self,idx = 0,regen = False,include_features = False):
		if not osp.exists(self.sample_file) or regen:
			self.generate_sample()
		sample_data = dd.io.load(self.sample_file)

		colors = ['grey', 'orangered']
		n_bins = 256  # Number of color gradients
		fire_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
		if not include_features:
			fig,ax = plt.subplots(2,2,figsize=(10,10),dpi=100)
			for axis in ax.flatten():
				axis.axis('off')
			ax[0,0].imshow(sample_data['data'][idx][-1],cmap= fire_cmap)
			ax[0,1].imshow(sample_data['label'][idx],cmap = fire_cmap)
			img = ax[1,0].imshow(sample_data['pred'][idx,0],cmap = 'jet')
			fig.colorbar(img,ax=ax[1,0])
			opt_threshold = self.best_threshold
			thresh_preds = (sample_data['pred'][idx,0] > opt_threshold).int()
			ax[1,1].imshow(thresh_preds,cmap = fire_cmap)

			ax[0,0].set_title('Previous Fire Mask')
			ax[0,1].set_title('True Fire Mask')
			ax[1,0].set_title('Predicted Fire Mask')
			ax[1,1].set_title('Predicted Fire Mask (Thresholded)')

			plt.savefig(osp.join(self.model_dir,'sample_plot_firemasks_only.png'),bbox_inches='tight')
			plt.show()

		