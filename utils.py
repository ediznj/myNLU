def tokenize(text):
	print('tokenize called')
	return text.split()

class LoadInfo:
	"""
	should create vocab (map_iter_dataset kind)
	record the total vocab
	consider unknown vocab, sos/eos, pad token
	tokenize the input with the supplied tokenizer 
	"""

	def __init__(self, inp_text, tokenizer=tokenize, **kwargs):
		self.__tokenizer = tokenizer
		self.__vocab = {}
		self.__kwargs = kwargs
		self.__vocab_list = []
		self.__data = inp_text
		self.__count = 0
		self.__build_vocab()
		self.__build_vocab(self.__data)
		self.__start=None

	def __build_vocab(self, iter_var=['unk sos eos pad']):
		#if token not exists then add
		#else return
		for i in iter_var:
			for token in self.__tokenizer(i):
				self.__add(token)

	def __add(self, token): #PRIVATE METHOD
		if not self.__vocab.get(token, None):
			self.__vocab[token] = self.__count
			self.__vocab_list.append(token)
			self.__count += 1

	def __len__(self):
		return self.__count

	def __getitem__(self, item):
		if type(item) == str:
			out = self.__vocab.get(item, None)
			return out if out else self.__vocab['unk']
		else:
			return self.__vocab_list[item]

	def __iter__(self):
		for sent in self.__data:
			yield [self.__vocab[text] 
				for text in self.__tokenizer(sent)
			]

	def size(self):
		return self.__len__()

	def yield_ids(self):
		for sent in self.__data:
			yield [
				self.__vocab[text]
				for text in self.__tokenizer(sent)
			]

	def yield_vocab(self):
		for voc in self.__vocab:
			yield voc

	def yield_
