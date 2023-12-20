import babelnet as bn
from babelnet import BabelSynsetID
from babelnet import Language
import argparse
import pandas as pd
import logging
import pickle
from collections import defaultdict

def get_args():

	parser = argparse.ArgumentParser(description='Mapping words with sense keys.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-target_set', default='targets-de', help='Name of target set', required=False,
						choices=['targets-de', 'targets-la', 'targets-sv'])
	args = parser.parse_args()
	return args

def load_target(target_path):
	target = pd.read_csv(target_path)
	return target

if __name__ == '__main__':

	args = get_args()
	'''
	Load target words.
	'''
	target_words = []
	target_path = 'data/%s.csv' % args.target_set
	targets = load_target(target_path)
	for t in targets['text']:
		target_words.append(t)
	logging.info('Finish loading target words.')

	'''
	Check candidate senses for each target word according to BabelNet.
	'''
	if args.target_set == 'targets-de':
		language = Language.DE
	elif args.target_set == 'targets-la':
		language = Language.LA
	elif args.target_set == 'targets-sv':
		language = Language.SV
	word2sense = defaultdict(list)

	embed_mapping_path = 'data/results/%s.pkl' %(args.target_set)
	with open(embed_mapping_path, 'wb') as mapping_f:
		for word in target_words:
			for synset in bn.get_synsets(word, from_langs=[language]):
				word2sense[word].append(synset.id)
		pickle.dump(word2sense, mapping_f)
	logging.info('Finish writing words to sense keys mapping.')

	# '''
	# Load word to sense mapping.
	# '''
	# with open(embed_mapping_path, 'rb') as mapping_f:
	# 	word2sense = pickle.load(mapping_f)
	# 	print(word2sense['aktiv'])