from transformers import BertModel, BertTokenizerFast

config_default = {
	'model': BertModel.from_pretrained('bert-base-uncased'),
	'tokenizer': BertTokenizerFast.from_pretrained('bert-base-uncased')
}


class BertEmbed:
	"""
	USAGE: bert_embed = BertEmbed().get_embedding(text)
		   intent_embed = bert_embed.get_intent_embedding()
		   slot_embed = bert_embed.get_slot_embedding()
	"""
	def __init__(self, **config):
		if not config: config = config_default
		self.bert = config.get('model',None)
        self.tokenizer = config.get('tokenizer',None)
        self.bert_out_dict = None
        self.embed_size = None

    def tokenize(self, text):
    	tokens = self.tokenizer(
    		text, 
    		return_attention_mask=True, 
    		return_token_type_ids=False
    	)
    	input_ids = torch.tensor(tokens.get('input_ids', None))
    	attn_mask = torch.tensor(tokens.get('attention_mask', None))
    	return input_ids, attn_mask

    def get_embedding(self, text):
    	self.bert_out_dict = bert(*self.tokenize(text), return_dict=True)
    	self.embed_size = self.bert_out_dict['pooler_output'].shape(-1)
    	return self.bert_out_dict, self.embed_size

    def get_intent_embedding(self):
    	return self.bert_out_dict['pooler_output']

    def get_slot_embedding(self):
    	return self.bert_out_dict['last_hidden_state']

