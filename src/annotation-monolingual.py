import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
import argparse
from time import time
from collections import defaultdict
import string
from functools import lru_cache
import math
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn_lemmatizer = WordNetLemmatizer()

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def get_args(
		batch_size = 64,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-test_set', default='ccoha1', help='Name of test set', required=False,
						choices=['ccoha1', 'ccoha2', 'c2'])
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('-sense_embeddings', type=str, default='ares', help='Pre-trained sense embeddings', required=False)
	parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
	parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
	parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
	parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
	parser.add_argument('-device', default='cuda', type=str)
	parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=True)
	args = parser.parse_args()
	return args

def load_corpus(ccoha_path):
	corpus = pd.read_csv(ccoha_path)
	return corpus


def load_target(target_path):
	target = pd.read_csv(target_path)
	return target

def get_id2sks(wsd_eval_keys):
	"""Maps ids of split set to sensekeys, just for in-code evaluation."""
	id2sks = {}
	with open(wsd_eval_keys) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			id2sks[id_] = keys
	return id2sks


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def str_scores(scores, n=3, r=5):  ###
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


@lru_cache()
def wn_first_sense(lemma):
	first_synset = wn.synsets(lemma)[0]
	found = False
	for lem in first_synset.lemmas():
		key = lem.key()
		if key.startswith('{}%'.format(lemma)):
			found = True
			break
	assert found
	return key


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def load_lmms(npz_vecs_path):
    lmms = {}
    loader = np.load(npz_vecs_path)
    labels = loader['labels'].tolist()
    vectors = loader['vectors']
    for label, vector in list(zip(labels, vectors)):
        lmms[label] = vector
    return lmms


def get_synonyms_sk(sensekey, word):
	synonyms_sk = []
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				for lemma2 in synset.lemmas():
					synonyms_sk.append(lemma2.key())
	return synonyms_sk


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT
	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]

	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]


def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def get_synonyms(sensekey, word):
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				synonyms_list = synset.lemma_names()
	return synonyms_list


def get_bert_embedding(sent):
	tok_text = tokenizer.tokenize(sent)
	tokenized_text = ["[CLS]"]+tok_text+["[SEP]"]
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0 for i in range(len(indexed_tokens))]
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = tokens_tensor.to(device)
	segments_tensors = segments_tensors.to(device)
	model.to(device)
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	layers_vecs = np.sum([outputs[2][-1].cpu().detach().numpy(), outputs[2][-2].cpu().detach().numpy(), outputs[2][-3].cpu().detach().numpy(), outputs[2][-4].cpu().detach().numpy()], axis=0) ### use the last 4 layers
	res = list(zip(tokenized_text[1:-1], layers_vecs[0][1:-1]))
	
	## merge subtokens
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
			merged_vec = torch.from_numpy(merged_vec)
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


@lru_cache()
def wn_lemmatize(w, postag=None):
    w = w.lower()
    return wn_lemmatizer.lemmatize(w)


class SensesVSM(object):

	def __init__(self, vecs_path, normalize=False, sense_embeddings='ares'):
		self.vecs_path = vecs_path
		self.labels = []
		self.matrix = []
		self.indices = {}
		self.ndims = 0
		self.sense_embedding = sense_embeddings

		if self.vecs_path.endswith('.txt'):
			if self.sense_embedding == 'ares':
				self.load_ares_txt(self.vecs_path)
			elif self.sense_embedding == 'lmms':
				self.load_txt(self.vecs_path)

		elif self.vecs_path.endswith('.npz'):
			self.load_npz(self.vecs_path)
		self.load_aux_senses()


	def load_txt(self, txt_vecs_path):
		self.vectors = []
		sense_vecs = {}
		with open(txt_vecs_path, encoding='utf-8') as vecs_f:
			for line_idx, line in enumerate(vecs_f):
				if line_idx == 0:
					continue
				elems = line.split()
				self.labels.append(elems[0])
				self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
				sense_vecs[self.labels] = np.array(list(map(float, elems[1:])), dtype=np.float32)
		self.vectors = np.vstack(self.vectors)

		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}

	
	def load_npz(self, npz_vecs_path):
		loader = np.load(npz_vecs_path)
		self.labels = loader['labels'].tolist()
		self.vectors = loader['vectors']

		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}
		self.ndims = self.vectors.shape[1]

	
	def load_ares_txt(self, path):
		self.vectors = []
		sense_vecs = {}
		with open(path, 'r') as sfile:
			for idx, line in enumerate(sfile):
				if idx == 0:
					continue
				splitLine = line.split(' ')
				self.label = splitLine[0]
				self.labels.append(self.label)
				self.vec = np.array(splitLine[1:], dtype='float32')
				self.vectors.append(self.vec)
				sense_vecs[self.label] = self.vectors
		self.vectors = np.vstack(self.vectors)
		self.indices = {l: i for i, l in enumerate(self.labels)}


	def load_aux_senses(self):
		self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
		self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

		self.lemma_sks = defaultdict(list)
		for sk, lemma in self.sk_lemmas.items():
			self.lemma_sks[lemma].append(sk)
		self.known_lemmas = set(self.lemma_sks.keys())

		self.sks_by_pos = defaultdict(list)
		for s in self.labels:
			self.sks_by_pos[self.sk_postags[s]].append(s)
		self.known_postags = set(self.sks_by_pos.keys())


	def match_senses(self, vec, lemma=None, topn=100):
		matches = []
		relevant_sks = []
		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
					relevant_sks.append(sk)
					context_vec = torch.cat((vec, vec), 0)

		relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]
		sims = np.dot(self.vectors[relevant_sks_idxs], np.array(context_vec))
		matches = list(zip(relevant_sks, sims))
		matches = sorted(matches, key=lambda x: x[1], reverse=True)	
		return matches[:topn]


if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	device = torch.device(args.device)

	'''
	Load pre-trianed sense embeddings for evaluation.
	Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
	Load fastText static embeddings if required.
	'''
	if args.sense_embeddings == 'lmms':
		embedding_path = '../bias-sense/data/lmms_2048.bert-large-cased.npz'
	elif args.sense_embeddings == 'ares':
		embedding_path = './senseEmbeddings/external/ares/ares_bert_large.txt'
	senses_vsm = SensesVSM(embedding_path, normalize=True, sense_embeddings='ares')
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()


	'''
	Initialize various counters for calculating supplementary metrics.
	'''
	n_instances, n_correct, n_unk_lemmas, acc_sum = 0, 0, 0, 0
	n_incorrect = 0
	num_options = []
	correct_idxs = []
	failed_by_pos = defaultdict(list)

	'''
	Load instances in ccoha1 and ccoha2, as well as target words.
	'''
	ccoha_path = 'data/%s.csv' % args.test_set
	corpus_insts = load_corpus(ccoha_path)
	logging.info('Finish loading texts from %s.' % args.test_set)

	target_words = []
	target_path = 'data/targets.csv'
	targets = load_target(target_path)
	for t in targets['text']:
		target_words.append(t)
	logging.info('Finish loading target words.')

	# instances = corpus_insts['text'].to_string(index = False)
	# instances_remove_punctuation = instances.translate(str.maketrans('', '', string.punctuation))
	# instances_remove_punctuation = pd.Series(instances_remove_punctuation)
	instances = corpus_insts['text']

	'''
	Iterate over instances and write predictions for each instance.
	'''
	sentece_count = 0
	total_count = 0
	results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.sense_embeddings)
	with open(results_path, 'w') as results_f:
		for batch_idx, batch in enumerate(chunks(instances, args.batch_size)):
			for sent_info in batch:	
				sent_remove_punctuation = sent_info.translate(str.maketrans('', '', string.punctuation))
				total_count +=1
				print("total_count: ", total_count)
				sent_bert = get_bert_embedding(sent_remove_punctuation)
				word2idx = defaultdict(list)
				idx = 0
				for i in sent_bert:
					word2idx[i[0]].append(idx)
					idx += 1
				for word in list(word2idx.keys()):
					if word not in target_words:
						continue
					sentece_count += 1
					curr_lemma =  wn_lemmatize(word)
					idxs = word2idx[word]
					currVec_c = torch.mean(torch.stack([sent_bert[i][1] for i in idxs]), dim=0)

					matches = []
					if curr_lemma not in senses_vsm.known_lemmas:
						n_unk_lemmas += 1
						matches = [(wn_first_sense(curr_lemma), 1)]
					else:
					matches = senses_vsm.match_senses(currVec_c, lemma=curr_lemma, topn=10)
					results_f.write('{}, {}, {}, {}\n'.format(sent_info, idxs, word, matches))
	logging.info('Num. unknown lemmas in %s: %d' % (args.test_set, n_unk_lemmas))
	logging.info('Num. annotated sentences in %s: %d' % (args.test_set, sentece_count))