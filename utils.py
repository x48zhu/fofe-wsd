import itertools
import logging
from os.path import join
import sys
import tensorflow as tf

def get_data_type(data_path):
	"""
	Get data type from the data path.
	Data path follows format: ROOT_PATH/DATA_TYPE/DATA_NAME.
	e.g. masc/semcor, omsti/Semcor
	"""
	return data_path.split('/')[-2]


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
    the output should be [0, 3, 7, 10]
    :param target_words: dict(str, TargetWord)
    :return: list(int),
    """
    accum_sense_idx = []
    sum_sense = 0
    accum_sense_idx.append(sum_sense)
    target_words_sorted_by_id = sorted(target_words.values(), key=lambda w: w.id)
    for i, target_word in enumerate(target_words_sorted_by_id):
        assert target_word.id == i, "Target word ID doesn't match"
        sum_sense += len(target_word.label2target)
        accum_sense_idx.append(sum_sense)
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

