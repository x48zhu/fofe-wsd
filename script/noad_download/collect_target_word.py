#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import argparse
import fnmatch
import json
import os
import time
import xml.etree.ElementTree as et

from collections import defaultdict
from os.path import join, isfile

from file_handler import *


def get_target_words_masc(dataset_path):
    """
    Get all target words from the datasets
    :param dataset_paths: list(str): paths for datasets
    :return: list(str): target words
    """
    target_words = {}
    for root_dir, dirnames, filenames in os.walk(dataset_path):
        for filename in fnmatch.filter(filenames, '*.xml'):
            # Loop through every xml file
            tree = et.parse(join(root_dir, filename))
            root = tree.getroot()
            parsed_xml = [child.attrib for child in root.findall('word')]
            for child in parsed_xml:
                if 'sense' in child:
                    if type(child['lemma']) is not str:
                        print child['lemma'], child['sense'], type(child['lemma']), "%s/%s/%s"%(root_dir, dirnames, filename)
                        continue
                    target_words[child['lemma']] = child['sense']
    print "Number of target words: ", len(target_words)
    return target_words


def main(args):
	semcor = get_target_words_masc(args.datasets + '/semcor')
        masc = get_target_words_masc(args.datasets + '/masc')
        semcor = set(semcor.keys())
        masc = set(masc.keys())
	diff = list(semcor ^ masc)
	union = list(semcor | masc)
	PlainTextFileHandler().write(diff, args.output_dir, 'target_words.diff')
	PlainTextFileHandler().write(union, args.output_dir, 'target_words.union')
	
	
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('datasets')
        parser.add_argument('output_dir')
        args, unparsed_args = parser.parse_known_args()

	main(args)

