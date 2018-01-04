import pickle
from os.path import join


class PickleFileHandler(object):
	def read(self, *args):
		return pickle.load(open(join(*(args)) + '.pkl', 'rb'))

	def write(self, data, *args):
		pickle.dump(data, open(join(*(args)) + '.pkl', 'wb'))

class PlainTextFileHandler(object):
	def read(self, *args):
		with open(join(*(args)) + '.txt', 'r') as f:
			return f.read().splitlines() 

	def write(self, lines, *args):
		with open(join(*(args)) + '.txt', 'w') as f:
			f.writelines(lines)

