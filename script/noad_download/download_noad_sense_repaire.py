#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import argparse

def main(args):
	train_data = PickleFileHandler().read(args.train_path)
	test_data = PickleFileHandler().read(args.test_path)
	results = PickleFileHandler().write(args.output_path, "mfs")
	for target_word in results:



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('train_path')
	parser.add_argument('test_path')
	parser.add_argument('output_path', help="Path to save predictions")
	args, unparsed_args = parser.parse_known_args()
	main(args)