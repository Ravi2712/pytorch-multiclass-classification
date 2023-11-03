mport copy
import json
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:4024"

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if cuda.is_available() else 'cpu'

import transformers
from transformers import RobertaModel, RobertaTokenizer, get_scheduler, AutoTokenizer

# torch.backends.cuda = True
# print(torch.cuda.get_allocator_backend())
# os.environ['backend'] = "max_split_size_mb:2024"
torch.cuda.set_per_process_memory_fraction(0.6)

##################
### Constants
##################

MAX_LEN = 512
BATCH_SIZE = 64

EPOCHS = 2
LEARNING_RATE = 1e-05

CLASSES = 6
MODEL_NAME = "roberta-base"

SEED = 91
DIM_SIZE = 768
DROPOUT = 0.1

RESULT_PATH = "./results/"

######################
### Model & Dataloader
######################

class RobertaCustom(torch.nn.Module):

	def __init__(self, freeze_base_layers=False):
		super(RobertaCustom, self).__init__()
		self.base_model = RobertaModel.from_pretrained(MODEL_NAME)
		self.classifier = torch.nn.Sequential(
			#torch.nn.Dropout(DROPOUT),
			#torch.nn.Linear(DIM_SIZE, DIM_SIZE),
			#torch.nn.ReLU(),
			torch.nn.Dropout(DROPOUT),
			torch.nn.Linear(DIM_SIZE, CLASSES)
		)

		if freeze_base_layers:
			for name, param in self.named_parameters():
				if 'classifier' not in name: # classifier layer
					param.requires_grad = False

	def forward(self, **inputs):

		# Extract inputs
		input_ids = inputs['input_ids']
		att_masks = inputs['attention_mask']
		#token_type_ids = inputs['token_type_ids']

		# pass through models
		model_output = self.base_model(input_ids=input_ids, attention_mask=att_masks)
		hidden_state = model_output.last_hidden_state[:, 0]
		#print("Hidden state dimension: {}".format(hidden_state.shape))
		#print("Slicing hidden-state dimension: {}".format(hidden_state[:, 0].shape))

		return self.classifier(hidden_state)

class CustomData(Dataset):

	def __init__(self, dataframe, tokenizer):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.text = dataframe.text
		self.targets = self.data.label

	def __len__(self):
		return len(self.text)

	def __getitem__(self, index):
		
		# TODO: Check here if padding arg to be set to MAX_LEN
		inputs = self.tokenizer.encode_plus(
			self.text[index],
			None,
			add_special_tokens=True,
			max_length=MAX_LEN,
			padding='max_length',
			return_token_type_ids=False,
			truncation=True
		)
		
		return {
			'input_ids': torch.tensor(inputs['input_ids'], device=device),
			'attention_mask': torch.tensor(inputs['attention_mask'], device=device),
			#'token_type_ids': torch.tensor(inputs["token_type_ids"]),
			'targets': torch.tensor(self.targets[index], device=device)
		}

######################
### Train/Valid/Plot
######################

def train(model, train_loader, optimizer, loss_function):

	# Training History
	training_history = {
		"running_loss": [],
		"running_acc": []
	}

	# Set model in train phase
	model.train()

	# Stats
	running_loss = 0.0
	running_corrects = 0
	running_acc = 0.0
	num_running_examples = 0

	# Start a complete batch progressbar
	total_examples = len(train_loader)*BATCH_SIZE
	progress_bar = tqdm(range(total_examples))

	for batch in train_loader:

		'''
		Reference: 
			- https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
			- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
			- https://huggingface.co/docs/transformers/training#train-in-native-pytorch
		'''
		# zero the parameter gradients
		optimizer.zero_grad()

		batch = {k: v for k, v in batch.items()}
		outputs = model(**batch)
		loss = loss_function(outputs, batch['targets'])

		# Predicted indicies
		_, preds = torch.max(outputs, dim=1)

		# Calculate the gradient and apply
		loss.backward()
		optimizer.step()
		#lr_scheduler.step()

		# observe model performance
		running_loss += loss.item()
		running_corrects += torch.sum(preds == batch['targets']).item()

		# Following metrics are in between epoch loss/accuracy (Let's call it as training-steps)
		num_running_examples += batch['targets'].size(0)
		training_history["running_loss"].append(running_loss/num_running_examples)
		training_history["running_acc"].append(running_corrects/num_running_examples)

		# Update the params
		progress_bar.update(len(outputs))

	# Clean up
	progress_bar.close()

	# Epochs
	epoch_train_loss = running_loss/len(train_loader)
	epoch_train_acc = running_corrects/total_examples
	training_history["epoch_loss"] = epoch_train_loss
	training_history["epoch_acc"] = epoch_train_acc

	print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))

	return training_history

def valid(model, test_loader, loss_function, best_test_acc=0.0, save_best_model=True, filepath="./roberta_model.pt"):

	# Valid phase
	model.eval()

	print("Validating model performance...")

	# Stats
	running_loss = 0.0
	running_corrects = 0

	# Valid Samples
	total_examples = len(test_loader)*BATCH_SIZE

	with torch.no_grad():

		for batch in test_loader:
			#.to(device, non_blocking=True)
			batch = {k: v for k, v in batch.items()}
			outputs = model(**batch)
			loss = loss_function(outputs, batch['targets'])

			# Predicted indicies
			_, preds = torch.max(outputs, dim=1)

			# observe model performance
			running_loss += loss.item()
			running_corrects += torch.sum(preds == batch['targets']).item() 

		# Epochs
		epoch_test_loss = running_loss/len(test_loader)
		epoch_test_acc = running_corrects/total_examples

		print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_test_loss, epoch_test_acc))

		# Save model based on validation accuracy
		# TODO: Maybe we need a decent metric to save model (Like, F1 score or whatever)
		if save_best_model and (epoch_test_acc > best_test_acc):
			best_model_wts = copy.deepcopy(model.state_dict())
			torch.save(best_model_wts, filepath)

	return {
		"epoch_loss": epoch_test_loss,
		"epoch_acc": epoch_test_acc
	}

##################################
### MAIN STARTS HERE
##################################

# Huggingface RoBERTa tokenizer
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME, truncation=True, do_lower_case=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True, do_lower_case=True)

# Load Train/Test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

# Create train/test data-loaders
training_set = CustomData(train_data, tokenizer)
testing_set = CustomData(test_data, tokenizer)

train_params = {
	'batch_size': BATCH_SIZE,
	'shuffle': True,
	'num_workers': 0,
	#'pin_memory':True
}

test_params = {
	'batch_size': BATCH_SIZE,
	'shuffle': False,
	'num_workers': 0,
	#'pin_memory':True
}

train_loader = DataLoader(training_set, **train_params)
test_loader = DataLoader(testing_set, **test_params)

# Intantiate Model/Optimizer/Loss
roberta_model = RobertaCustom()
roberta_model.to(device, non_blocking=True)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(params=roberta_model.parameters(), lr=LEARNING_RATE)

# Torch train loop
best_test_acc = -1

for e_idx in range(EPOCHS):

	print('-' * 10)
	print('Epoch {}/{}'.format(e_idx, EPOCHS-1))
	print('-' * 10, '\n')

	# Train & Valid
	train_history = train(roberta_model, train_loader, optimizer, loss_function)
	test_history = valid(roberta_model, test_loader, loss_function, best_test_acc, filepath=RESULT_PATH+"roberta_model.pt")

	# Best validation accuracy
	if test_history["epoch_acc"] > best_test_acc:
		best_test_acc = test_history["epoch_acc"]
		print("Saving model at epoch-{} with best validation accuracy {:.4f}".format(e_idx, best_test_acc))
