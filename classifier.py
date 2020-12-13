import torch.nn as nn


class IntentClassifier(nn.Module):
	def __init__(self, intent_embed_size=None, intent_classes_size=None):
		assert intent_embed_size and intent_classes_size, 'IntentClassifier:CHECK YOUR INPUTS'
		super().__init__(self)
		inDim = intent_embed_size
		outDim = intent_classes_size
		self.out = nn.Linear(inDim, outDim)

	def forward(self,x):
		x = self.out(x)
		return x


class SlotClassifier(nn.Module):
	def __init__(self, slot_embed_size, slot_classes_size):
		assert slot_embed_size and slot_classes_size, 'SlotClassifier: CHECK YOUR INPUTS'
		super().__init__(self)
		inDim = slot_embed_size
		outDim = slot_classes_size
		self.out = nn.Linear(inDim, outDim)

	def forward(self):
		x = self.out(x)
		return x

	def train_slot(self, text)