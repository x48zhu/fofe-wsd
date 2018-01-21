#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import argparse
import fnmatch
import requests
import json
import os
import time
import xml.etree.ElementTree as et

from collections import defaultdict
from os.path import join, isfile

from file_handler import *

app_id = 'ee13b24d'
app_key = '12e64753882efec95ae02ffcbd4dc7a3'


def get_target_words_masc(dataset_path, target_words):
    """
    Get all target words from the datasets
    :param dataset_paths: list(str): paths for datasets
    :return: list(str): target words
    """
    for root_dir, dirnames, filenames in os.walk(dataset_path):
        for filename in fnmatch.filter(filenames, '*.xml'):
            # Loop through every xml file
            tree = et.parse(join(root_dir, filename))
            root = tree.getroot()
            parsed_xml = [child.attrib for child in root.findall('word')]
            for child in parsed_xml:
                if 'sense' in child:
                    target_words[child['lemma']] = True
    print "Number of target words: ", len(target_words)


def retrieve_sense_id(senseIds):
    for senseId in senseIds:
        if senseId.startswith('m_en_'):
            return senseId
    return None


def download_example_sentences(target_words):
    """
    Download example sentence for target words
    :param target_words: list(str): target words
    :return: dict(str, list(str)): examples for target words
    """
    _, sense_map_new2old = load_sense_map()
    examples_for_all_words = defaultdict(dict)

    unavaliable_entries = []

    for i, target_word in enumerate(target_words[:500]):
        # Accomodate 60 request limit to NOAD
        if i % 60 == 1:
            time.sleep(60)
        try:
            url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/en/' + target_word.lower() + '/sentences'
            r = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})

            for result in r.json()["results"]:
                lexicalEntries = result["lexicalEntries"]
                for lexicalEntry in lexicalEntries:
                    sentences = lexicalEntry["sentences"]
                    for sent in sentences:
                        sense_id = retrieve_sense_id(sent["senseIds"])
                        if sense_id is None:
                            unavaliable_entries.append((target_word, sent))
                            print "### ", sent["senseIds"]
                            raise Exception('No valid sense id')
                        if sense_id not in sense_map_new2old:
                            print "### %s has sense id %s not found" % (target_word, sense_id)
                            raise Exception('New sense id not found')
                        old_sense_id = sense_map_new2old[sense_id]
                        text = sent["text"]
                        examples = examples_for_all_words[target_word]
                        if old_sense_id not in examples:
                            examples[old_sense_id] = []
                        examples[old_sense_id].append(text)
        except Exception as e:
            print '## Download examples for %s failed' % target_word
            print e
    return examples_for_all_words, unavaliable_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets')
    parser.add_argument('output')
    args, unparsed_args = parser.parse_known_args()

    try:
        target_words = PickleFileHandler().read(args.output, 'target_word.vocab')
    except IOError:
        target_words = {}
        get_target_words_masc(args.datasets + '/semcor', target_words)
        get_target_words_masc(args.datasets + '/masc', target_words)
        PickleFileHandler().write(target_words.keys(), args.output, 'target_word.vocab')

    examples_for_all_words, unavaliable_entries = download_example_sentences(target_words)
    if len(unavaliable_entries) > 0:
        PickleFileHandler().write(unavaliable_entries, args.output, 'unavaliable_entries')    
    PickleFileHandler().write(examples_for_all_words, args.output, 'raw_noad.data')
    print "Total target words covered: %d" % len(examples_for_all_words)

