#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import argparse
from os.path import basename

from data_constructor import process_data
from file_handler import PickleFileHandler
from language_model import *
from utils import *


def main(args):
	logger.info(args)
	logger.info("Load word embedding...")
	word2idx = load_word_list(args.wordlist_path)

	logger.info("Process data...")
	target_words = {}
	data_constructor = process_data(args.data_path, args.data_type, word2idx, target_words, args.context_size)
	logger.debug("Number of target words: %d" % len(target_words))
	logger.info("Save process data...")
	PickleFileHandler().write((target_words, data_constructor), args.output_dir, "%s_%s.data_constructor" % (args.data_type, basename(args.data_path)))
	logger.info("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('wordlist_path', help="path to word list file")
	parser.add_argument('data_type', help="data type (masc/omsti/noad)")
	parser.add_argument('data_path', help="path to data")
	parser.add_argument('output_dir')

	parser.add_argument('--context_size', type=int, default=16)
	args, unparsed_args = parser.parse_known_args()
	main(args)
