import torch
import torch.nn as nn

import pretrained
import utils
from classifier import IntentClassifier, SlotClassifier

# class SimpleBert:
# 	def __init__(self, texts):
# 		self.texts = texts
# 		self.embed_size

# 	def get_embed(self):
# 		data = utils.LoadInfo(self.texts)
# 		embed_dict, self.embed_size = pretrained.BertEmbed().get_embedding(data)
# 		return embed_dict, self.embed_size

class BertNLU:
	"""
	USAGE: nlu = BertNLU(text)
		   ic = nlu.intent_classifier()
		   sc = nlu.slot_classifier()
	"""
	def __init__(self, texts, intents=None, slots=None):
		self.intents = utils.LoadInfo(intents)
		self.intent_embed = None
		self.slots = utils.LoadInfo(slots)
		self.slot_embed = None

		self.embed_size = None
		self.get_embed(texts)
	
	def get_embed(self, text):
		text = utils.LoadInfo(text)#seq_in
		embed, self.embed_size = pretrained.BertEmbed().get_embedding(text)
		self.intent_embed = embed.get_intent_embedding()
		self.slot_embed = embed.get_slot_embedding()

	def intent_classifier(self):
		return IntentClassifier(
			self.embed_size,
			self.intents.size()
		)

	def slot_classifier(self):
		return SlotClassifier(
			self.embed_size,
			self.slots.size()
		)

	def intent_ids(self):
		return intents.ids()

	def slot_ids(self):
		return slots.ids()

