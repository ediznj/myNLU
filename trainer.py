# trainer.py
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


"""

"""

class Trainer(object):
	def __init__(self):
		pass

	def train(self):
		pass

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass


def train(name=None, **kwargs):

	if kwargs:
		model = kwargs['model']
		loader = kwargs['inp_args']
		criterion = kwargs['criterion']
		optim = kwargs['optim']
		expected = kwargs['expected']
		num_epoch = kwargs['epoch']
	else:
		return None

	for epoch in range(num_epoch):
		for inp in loader:
			out = model_ic(inp)
			loss = criterion(out, expected)
			optim.zero_grad()
			loss.backward()
			optim.step()
	return out