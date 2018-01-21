#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import argparse
from os.path import basename

from data_constructor import process_data
from file_handler import PickleFileHandler
from language_model import *
from utils import *


def main(args):
	logger.info(args)

	if args.processed:
		pkl_obj = PickleFileHandler().read(args.output_dir,  "%s_%s.data_constructor" % (args.data_type, basename(args.data_path)))
		target_words, data_constructor = pkl_obj[0], pkl_obj[1]
	else:
		logger.info("Load word embedding...")
		word2idx = load_word_list(args.wordlist_path)

		logger.info("Process data...")
		target_words = {}
		data_constructor = process_data(args.data_path, args.data_type, word2idx, target_words, args.context_size, args.ngram)
		logger.debug("Number of target words: %d" % len(target_words))

	logger.info("Load language model...")
	language_model = load_language_model(args.model_path, args.alpha)
	
	logger.info("Abstract context...")
	context = {}
	for target_word in target_words.values():
		features = []
		labels = []
		poss = []
		for target, label in target_word.target2label.items():
			for left_context, right_context, pos in data_constructor.next_by_target(args.batch_size, target_word.word, target, ngram=args.ngram):
				features.extend(language_model.infer(left_context, right_context))
				labels.extend([label] * left_context.shape[0])
				poss.extend(pos)
		context[target_word.word] = (features, labels, poss)
	
	logger.info("Save abstracted context...")
	PickleFileHandler().write(
		context, args.output_dir, '%s_%s_%s_%s' % (basename(args.model_path), args.data_type, basename(args.data_path), args.desc)
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path', help="path to pretrained language model")
	parser.add_argument('wordlist_path', help="path to word list file")
	parser.add_argument('data_type', help="data type (masc/omsti/noad)")
	parser.add_argument('data_path', help="path to data")
	parser.add_argument('desc', help="description of current abstraction job")
	parser.add_argument('alpha', type=float)
	parser.add_argument('embedding_size', type=int)
	parser.add_argument('output_dir')

	parser.add_argument('--processed', dest='processed', action='store_true')
	parser.set_defaults(processed=False)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--context_size', type=int, default=16)
	parser.add_argument('--ngram', type=int, default=2)
	args, unparsed_args = parser.parse_known_args()
	main(args)

