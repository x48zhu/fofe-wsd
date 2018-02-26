import itertools
import logging
from os.path import join
import sys
import tensorflow as tf


app_id = 'ee13b24d'
app_key = '12e64753882efec95ae02ffcbd4dc7a3'
lexid_map_path = '/local/scratch/xzhu/work/lexid_map.csv'
wn_noad_map_dir = '/eecs/research/asr/xzhu/work/fofe-wsd/data/masc'


def map_pos(pos):
    the_map = {
        'noun':'n',
        'adjective':'a',
        'adj':'a',
        'verb':'v',
        'adv':'r',
        'adverb':'r',
        # 'pron':'pron',
        # 'adp':'adp',
        # 'conj':'conj',
        # 'det':'det',
        # 'num':'num',
        # 'prt':'prt',
        # 'x':'x',
        # 'other':'x',
        # 'preposition':'apd',
        # 'abbreviation':'x',
    }
    return the_map.get(pos.lower(), None)


def get_data_type(data_path):
	"""
	Get data type from the data path.
	Data path follows format: ROOT_PATH/DATA_TYPE/DATA_NAME.
	e.g. masc/semcor, omsti/Semcor
	"""
	return data_path.split('/')[-2]


def load_sense_map():
    # line[0] = m_en_gb; line[1] = m_en_gbus
    sense_map_old2new = {}
    sense_map_new2old = {}
    with open(lexid_map_path, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split(',')
            sense_map_new2old[temp[0]] = temp[1]
            sense_map_old2new[temp[1]] = temp[0]
    return sense_map_new2old, sense_map_old2new


def load_wn_noad_map():
    wn2noad = {}
    with open(join(wn_noad_map_dir, 'manual_map.txt'), 'r') as f:
        for line in f.readlines():
            temp = line.strip().split('\t')
            noad = temp[0].split('/')[-1]
            wns = temp[1]
            for wn in wns.split(','):
                wn2noad[wn] = noad
    return wn2noad


def map_fn_wrapper(fn, arrays, name, dtype=tf.float32):
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]),
                    indices, dtype=dtype, name=name)
    return out


def calculate_accumulate_sense_index(target_words):
    """
    Target words have different number of senses. We want a unified sense index based on these senses for all
    target words.
    For example, target words 'a','b','c' each have 3,4,3 sense, this function should give these 3+4+3=10 senses, then
    the output should be {'a':0, 'b':4, 'c':9, 'END':13]
    :param target_words: dict(str, TargetWord)
    :return: list(int),
    """
    accum_sense_idx = {}
    sum_sense = 0
    accum_sense_idx.append(sum_sense)
    for target_word in target_words.value(): 
        accum_sense_idx[target_word.word] = sum_sense
        sum_sense += len(target_word.label2target) + 1 # plus one for unknown
    accum_sense_idx['END'] = sum_sense
    return accum_sense_idx


def concat_lists(list_of_lists):
    """
    Concatenate list of lists
    :param list_of_lists: list(list)
    :return: list
    """
    return list(itertools.chain.from_iterable(list_of_lists))


def harmonic_mean(a, b):
    return 2.0 * (a * b) / (a + b)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

