#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import argparse
from os.path import basename

from data_constructor import process_data
from file_handler import PickleFileHandler
from language_model import *
from utils import *


def main(args):
	logger.info(args)

	logger.info("Load language model and word embedding...")
	language_model = load_language_model(args.model_path, args.alpha)
	word2idx = load_word_list(args.wordlist_path)

	logger.info("Process data...")
	target_words = {}
	data_constructor = process_data(args.data_path, args.data_type, word2idx, target_words, args.context_size)
	logger.debug("Number of target words: %d" % len(target_words))
	
	logger.info("Abstract context...")
	context = {}
	for target_word in target_words.values():
		features = []
		labels = []
		# for target in sorted(target_word.target2label.keys()):
		for target, label in target_word.target2label.items():
			for left_context, right_context in data_constructor.next_by_target(args.batch_size, target_word.word, target):
				features.extend(language_model.infer(left_context, right_context))
				labels.extend([label] * left_context.shape[0])
		context[target_word.word] = (features, labels)
	
	logger.info("Save abstracted context...")
	PickleFileHandler().write(
		context, args.output_dir, '%s_%s_%s' % (basename(args.model_path), args.data_type, basename(args.data_path))
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path', help="path to pretrained language model")
	parser.add_argument('wordlist_path', help="path to word list file")
	parser.add_argument('data_type', help="data type (masc/omsti/noad)")
	parser.add_argument('data_path', help="path to data")
	parser.add_argument('alpha', type=float)
	parser.add_argument('embedding_size', type=int)
	parser.add_argument('output_dir')

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--context_size', type=int, default=16)
	args, unparsed_args = parser.parse_known_args()
	main(args)
