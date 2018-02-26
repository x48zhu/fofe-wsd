#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import argparse
import json
import requests
import time

from file_handler import PickleFileHandler


app_id = 'ee13b24d'
app_key = '12e64753882efec95ae02ffcbd4dc7a3'

app_id_keys = [
	('8f15a4ff', '34d1c7a4a2b061e03d0596309e94ceec'),
	('9909883b', 'cc26e11ffce7b34564e05119c2513bc3'),
	('9235e70c', 'fb5c32eb34cdee7cb762636dc5c22a33'),
	('f1dfae01', '218f5f6291d15029424d60b98a04a103'),
	('3eb07e12', 'fdbb470f7bcd35437f19496882b5ab31'),
	('3492e2a9', '22a6f9e119d1ff0d8f19b5b95a0318d9'),
	('14fb7111', '3e009970aaa3d2068125a4f2ec43e74d'),
]

lexid_map_path = '/local/scratch/xzhu/work/lexid_map.csv'


lexical_category_map = {
  'ADJ'  :         'Adjective',
  'ADP'  :         'Adposition',
  'ADV'  :         'Adverb',
  'CONJ' :         'Conjunction',
  'DET'  :         'Determiner',
  'NOUN' :         'Noun',
  'NUM'  :         'Numeral',
  'PRON' :         'Pronoun',
  'PRT'  :         'Particle',
  'VERB' :         'Verb',
  'X'    :         'Other',
}

word_map = {
  'maneuver':'manoeuvre',
}


class Requester:
	def __init__(self, id_keys=app_id_keys):
		self.id_keys_idx = 0
		self.id_keys = id_keys
		self.app_id = self.id_keys[self.id_keys_idx][0]
		self.app_key = self.id_keys[self.id_keys_idx][1]
		self.total_count = 0
		self.fail_instance = []

	def fail_handler(self, target_word, pos, error, url=None):
		print "### [%s, %s] " % (target_word, pos) + error
		self.fail_instance.append((target_word, pos))
		if url is not None:
			print url

	def most_freq_sense_masc(self, target_word, pos):
		self.total_count += 1
		if self.total_count % 60 == 0:
			time.sleep(60)
		if pos not in lexical_category_map:
			self.fail_handler(target_word, pos, "has invalid lexical category")
			return None
		pos = lexical_category_map[pos]
		target_word_lower = target_word.lower()
		url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/en/' + word_map.get(target_word_lower, target_word_lower) + '/definitions;lexicalCategory=' + pos.lower()
		r = requests.get(url, headers = {'app_id': self.app_id, 'app_key': self.app_key})
			
		if r.status_code == 403:
			print "App id %s used up." % self.app_id
			self.id_keys_idx += 1
			self.app_id = self.id_keys[self.id_keys_idx][0]
			self.app_key = self.id_keys[self.id_keys_idx][1]
		elif r.status_code > 200:
			self.fail_handler(target_word, pos, "request returns status_code %d" % r.status_code, url)
			return None

		try:
			new_id = r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['id']
		except:
			self.fail_handler(target_word, pos, "result parse failed", url)
			return None
		if new_id in sense_map_new2old:
			old_id = sense_map_new2old[new_id]
		else:
			self.fail_handler(target_word, pos, "cannot be found in NOAD dictionary map", url)
			old_id = None
		return old_id

	def summary(self, output_path):
		print "%d success, %d fail" % (self.total_count, len(self.fail_instance))
		PickleFileHandler().write(results, output_path, "failed_instance")


def load_sense_map():
	# line[0] = m_en_gb; line[1] = m_en_gbus
	sense_map_old2new = {}
	sense_map_new2old = {}
	with open(lexid_map_path, 'r') as f:
		for line in f.readlines():
			temp = line.strip().split(',')
			old = temp[0]
			new = temp[1]
			sense_map_new2old[new] = old
			sense_map_old2new[old] = new
	return sense_map_new2old, sense_map_old2new


sense_map_new2old, _ = load_sense_map()


def main(args):
	print(args)
	train_data = PickleFileHandler().read(args.train_path)
	test_data = PickleFileHandler().read(args.test_path)
	results = {}
	requester = Requester()

	for target_word in train_data.keys():
		results[target_word] = {}
		labels = train_data[target_word][1]
		poss = train_data[target_word][2]
		for pos in set(poss):
			results[target_word][pos] = None

	for target_word in test_data.keys():
		if target_word not in results:
			results[target_word] = {}
		labels = test_data[target_word][1]
		poss = test_data[target_word][2]
		for pos in set(poss):
			results[target_word][pos] = None
	print "%d target word" % len(results)

	i = 0
	for target_word in results:
		for pos in results[target_word]:
			i += 1
			if i % 60 == 0:
				time.sleep(60)
			results[target_word][pos] = requester.most_freq_sense_masc(target_word, pos)

	PickleFileHandler().write(results, args.output_path, "mfs")
	requester.summary(args.output_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('train_path')
	parser.add_argument('test_path')
	parser.add_argument('output_path', help="Optional, path to save predictions")
	args, unparsed_args = parser.parse_known_args()
	main(args)
