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

language = 'en'
word_id = 'bank'

url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/' + word_id.lower() + '/grammaticalFeatures%3Dsingular%2Cpast%3BlexicalCategory%3Dnoun%3Bdefinitions'

r = requests.get(url, headers = {'app_id': app_id, 'app_key': app_key})

print("code {}\n".format(r.status_code))
print("text \n" + r.text)
print("json \n" + json.dumps(r.json()))

