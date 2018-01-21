#!/eecs/home/xzhu/xzhu_env/bin/python
import argparse
import fnmatch
import requests
import json
import os
import time
import sys
import xml.etree.ElementTree as et

from collections import defaultdict
from os.path import join, isfile

from file_handler import *
from utils import *

app_id = 'ee13b24d'
app_key = '12e64753882efec95ae02ffcbd4dc7a3'

app_id = '1af6076e'
app_key = '98699251c49a5bdc6eed8a4a6a15810f'
lexid_map_path = '/local/scratch/xzhu/work/lexid_map.csv' # TODO

def process(args):
    raw_results = PickleFileHandler().read(args.output_dir, 'raw_dict')
    sense_map_new2old, _ = load_sense_map()
    mfs_results = {}
    for word, content in raw_results.items():
        try:
            mfs_ids = {}
            for lexical_entry in content[0]['lexicalEntries']:
                pos = map_pos(lexical_entry['lexicalCategory'])
                if pos is None:
                    continue
                entries = lexical_entry['entries']
                mfs_new_id = entries[0]['senses'][0]['id']
                for sense in entries[0]['senses']:
                    mfs_new_id = sense['id']
                    if mfs_new_id in sense_map_new2old:
                        mfs_ids[pos] = sense_map_new2old[mfs_new_id]
                        break
                    print word, "No available sense id"
            mfs_results[word] = mfs_ids
        except Exception as e:
            print word, e
    PickleFileHandler().write(mfs_results, args.output_dir, 'mfs.dict_entries')


def download(args):
    results = {}
    word_list = PlainTextFileHandler().read(args.word_list_path)
    for i, word in enumerate(word_list):
        url = "https://od-api.oxforddictionaries.com:443/api/v1/entries/en/" + word + "/regions=us"
        try:
            r = requests.get(url, headers = {'app_id': app_id, 'app_key': app_key})
            if not r.status_code == 200:
                raise "Bad response"
            results[word] = r.json()['results']
            if i % 50 == 0:
                print "Processed %d words" % (i + 1)
                time.sleep(60)
        except Exception as e:
            print word, e, url
    PickleFileHandler().write(results, args.output_dir, 'raw_dict')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('word_list_path', help="path to list of word to download sense")
    parser.add_argument('output_dir')
    parser.add_argument('task')
    args, unknown_args = parser.parse_known_args()

    if args.task == "download":
        download(args)
    elif args.task == "process":
        process(args)
    else:
        raise "Invalid task type"


