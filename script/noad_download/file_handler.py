import pickle
from os.path import join


def join_append_extension_if_not_exist(extension, *args):
	if not extension.startswith('.'):
		extension = '.' + extension
	args = args[0]
	if not args[-1].endswith(extension):
		args = list(args)
		args[-1] = args[-1] + extension
	return join(*(args))


class PickleFileHandler(object):
	def read(self, *args):
		fname = join_append_extension_if_not_exist('pkl', args)
		return pickle.load(open(fname, 'rb'))

	def write(self, data, *args):
		fname = join_append_extension_if_not_exist('pkl', args)
		pickle.dump(data, open(join(*(args)) + '.pkl', 'wb'))

class PlainTextFileHandler(object):
	def read(self, *args):
		fname = join_append_extension_if_not_exist('txt', args)
		with open(fname, 'r') as f:
			return f.read().splitlines() 

	def write(self, lines, *args):
		fname = join_append_extension_if_not_exist('txt', args)
		with open(fname, 'w') as f:
			f.write('\n'.join(lines))

