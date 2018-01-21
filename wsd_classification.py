#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import argparse
from nltk.corpus import wordnet as wn
from sklearn import preprocessing
from sklearn.metrics import f1_score

from file_handler import PickleFileHandler
from classifier import *
from utils import *


class ClassifierGenerator(object):
	def __init__(self, method, options):
		self.method = method
		self.options = options

	def new_classifier(self, local_options):
		if self.method == "knn":
			k = min(float(self.options), local_options['k'])
			classifier = KNNClassifier(k, local_options['cos'])
		elif self.method == "avg":
			classifier = AverageClassifier(local_options['cos'])
		elif self.method == "mfs":
			classifier = MostFreqClassifier()
		elif self.method == "nn":
			classifier = NNClassifier()
		else:
			raise "Unknown classification method"
		return classifier


class LabelEncoder(object):
	def fit(self, labels):
		label_set = set(labels)
		self.label2target = {}
		self.target2label = {}
		for i, label in enumerate(label_set):
			self.label2target[label] = i + 1
			self.target2label[i + 1] = label

	def transform(self, labels):
		return map(lambda l: self.label2target.get(l, 0), labels)

	def inverse_transform(self, targets):
		return map(lambda t: self.target2label[t], targets)


def most_freq_sense_masc(target_word, pos=None):
	if pos is not None:
		synsets = wn.synsets(target_word, map_pos(pos))
	else:
		synsets = wn.synsets(target_word)
	for synset in synsets:
		for lemma in synset.lemmas(): # Find this lemma in the synset
			key = lemma.key()
			if key.startswith("%s" % target_word):
				return key
	return None


def main(args):
	logger.info(args)
	data_type = args.train_path.split('/')[-1].split('_')[1]
	if data_type == "masc":
		# Hardcode path
		oxford_dict = PickleFileHandler().read('/local/scratch/xzhu/work/fofe-wsd/data/noad/mfs.dict_entries')

	train_data = PickleFileHandler().read(args.train_path)
	test_data = PickleFileHandler().read(args.test_path)
	classifier_generator = ClassifierGenerator(args.method, args.options)

	classifiers = {}
	label_encoders = {}
	label_offsets = {}
	logger.info("Create classifiers...")
	for target_word in train_data.keys():
		features = train_data[target_word][0]
		labels = train_data[target_word][1]
		le = LabelEncoder()
		le.fit(labels)
		label_encoders[target_word] = le
		classifier = classifier_generator.new_classifier({'k':len(features), 'cos':args.cos})
		classifier.fit(np.array(features), np.array(le.transform(labels)))
		classifiers[target_word] = classifier
		label_offsets[target_word] = len(set(labels)) + 1

	# Evaluate
	ok = 0
	notok = 0
	total = 0
	agg_targets = []
	agg_predictions = []
	logger.info("Evaluate...")
	for target_word in test_data.keys():
		features = test_data[target_word][0]
		labels = test_data[target_word][1]
		poss = test_data[target_word][2]
		n = len(features)
		if target_word in classifiers:
			le = label_encoders[target_word]
			targets = le.transform(labels)
			predictions = classifiers[target_word].predict(np.array(features))
		elif args.mfs:
			le = LabelEncoder()
			le.fit(labels)
			targets = le.transform(labels)
			predictions = []
			for i, pos in enumerate(poss):
				if data_type == "omsti":
					predictions = le.transform(
						[most_freq_sense_masc(target_word, pos) for pos in poss]
					)
				elif data_type == "masc":
					predictions = le.transform(
						[oxford_dict.get(target_word, {}).get(map_pos(pos), 0) for pos in poss]
					)
				else:
					raise "Invalid data type"
		else:
			# No mfs backup, all predictions are false
			agg_targets.extend([0] * n)
			agg_predictions.extend([1] * n)
		n_correct = np.sum(np.equal(targets, predictions))
		ok += n_correct
		notok += np.size(targets) - n_correct
		total += n
	logger.info("Sklearn micro F1 score: %f" % f1_score(agg_targets, agg_predictions, average='micro'))
	precision = ok / float(ok + notok)
	recall = ok / float(total)
	if precision + recall == 0:
		f1 = 0
	else:
		f1 = (2 * precision * recall) / (precision + recall)
	logger.info("Evaluation result:")
	logger.info("  Precision: %f" % precision)
	logger.info("  Recall:    %f" % recall)
	logger.info("  F1:        %f" % f1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('train_path')
	parser.add_argument('test_path')
	parser.add_argument('method')
	parser.add_argument('--options', type=str)
	parser.add_argument('--mfs', dest='mfs', action='store_true')
	parser.set_defaults(mfs=False)
	# parser.add_argument('--cos', dest='cos', action='store_true')
	parser.set_defaults(cos=True)
	parser.add_argument('--output_path', help="Optional, path to save predictions")
	args, unparsed_args = parser.parse_known_args()
	main(args)
