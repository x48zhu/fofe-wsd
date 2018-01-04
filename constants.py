import numpy as np
import tensorflow as tf
from os.path import join

bos = "<s>"
eos = "</s>"
w_unknown = '<unk>'
pad = '<pad>'

breaklevel2idx = {
    'NO_BREAK': 0,
    'SPACE_BREAK': 1,
    'PARAGRAPH_BREAK': 2,
    'LINE_BREAK': 2,
    'SENTENCE_BREAK': 3
}

mixed_str = ""

FLAGS = None

test_target_words = [
               # Semcor   MASC
    # "day",     # 318, 2;  406 2
    "important",
    "education",
    # "place",   # 295 6;   199 6
    # "show",    # 316 6;   228 5
    # "play",    # 241 8;   136 8
    # "fall",    # 116 9;   138 9
    # "stress",  # 113 4;   11 3
    # "stock",   # 38 5;    34 3
    # "blood",   # 38 2;    44 2
    # "bank"     # 37 5;    79 3
]

poss = {
  '.':         	'Punctuation',
  'ADJ':        'Adjectives',
  'ADP':       	'Adpositions (prepositions and postpositions)',
  'ADV':       	'Adverbs',
  'CONJ':      	'Conjunctions',
  'DET':       	'Determiners',
  'NOUN':      	'Nouns (common and proper)',
  'NUM':       	'Cardinal numbers',
  'PRON':      	'Pronouns',
  'PRT':       	'Particles or other function words',
  'VERB':      	'Verbs (all tenses and modes)',
  'X':         	'Other: foreign words, typos, abbreviations'
}
