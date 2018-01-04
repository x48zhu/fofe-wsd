import xml.etree.ElementTree as et
from lxml import html

from constants import *
from utils import *
from data import *

def normalize(word):
    """
    Normalization that applies to every word in a document
    :param word: str, original word
    :return: str, normalized word
    """
    return word.lower()


def augment_data(origin_data, num):
    """
    Data augmentation
    :param origin_data:
    :param num: number of data to augment
    :return: a list of augmented data
    """
    return np.random.choice(origin_data, num, replace=True).tolist()


def extract_words_masc(dataset_path, word2idx, target_words, context_size):
    """
    Extract context words and target words from xml files of given data set. It does following jobs:

    1. Create object for all target words
    2. Extract all target word data, and related it with target word in 1
    3. Convert document into index and save to list

    :param parsed_xml: list(dict(str, str)), a list of dictionary that
        contains attributes of a word
    :param word2idx: dict(str, int), mapping word to index
    :param context_size
    :param target_words
    :param doc_indexed
    :param data_group_by_word
    :return: None
    """

    data_group_by_word = defaultdict(list)
    indexed_docs = []
    padding = np.empty(context_size, dtype=np.int32)
    padding.fill(word2idx[pad])
    w_unknown_idx = word2idx[w_unknown]

    for root_dir, dirnames, filenames in os.walk(dataset_path):
        for filename in fnmatch.filter(filenames, '*.xml'):
            # Loop through every xml file
            tree = et.parse(join(root_dir, filename))
            root = tree.getroot()
            parsed_xml = [child.attrib for child in root.findall('word')]

            # Initialization
            n_doc = len(indexed_docs)
            doc_idxs = []

            # Master before stash apply work barnch
            # doc_idxs.append(word2idx[bos])
            # for position, child in enumerate(parsed_xml):

            doc_idxs.append(word2idx[bos])
            position = 0
            for child in parsed_xml:
                position += 1 # add one first since bos append above
                text = normalize(child['text'])
                word_idx = word2idx.get(text, w_unknown_idx)

                break_level_idx = child['break_level']
                # if break_level_idx in ['SENTENCE_BREAK', 'PARAGRAPH_BREAK', 'LINE_BREAK']:
                #     doc_idxs.append(word2idx[eos])
                #     doc_idxs.append(word2idx[bos])
                doc_idxs.append(word_idx)

                if 'sense' in child:
                    # Encounter a target word data
                    lemma = child['lemma']
                    pos = child['pos']
                    if lemma not in target_words:
                        target_word = TargetWord(lemma)
                        target_words[lemma] = target_word
                    target_word = target_words[lemma]
                    label = child['sense'].split('/')[-1]
                    # label = int(child['sense'].split('.')[1])
                    if label not in target_word.label2target:
                        sense_idx = len(target_word.label2target)
                        target_word.label2target[label] = sense_idx
                        target_word.target2label[sense_idx] = label
                    target = target_word.label2target[label]
                    data = Data(target_word, n_doc, position, pos, target)
                    data_group_by_word[lemma].append(data)
            # doc_idxs.append(word2idx[eos])
            doc_idxs = np.concatenate([padding, np.asarray(doc_idxs, dtype=np.int32), padding], 0)
            indexed_docs.append(doc_idxs)
    return indexed_docs, data_group_by_word


def extract_words_omsti(dataset_path, word2idx, target_words, context_size):
    """
    Extract context words and target words from the given parsed xml data

    1. Create object for all target words
    2. Extract all target word data, and related it with target word in 1
    3. Convert document into index and save to list

    :param parsed_xml: list(dict(str, str)), a list of dictionary that
        contains attributes of a word
    :param word2idx: dict(str, int), mapping word to index
    :param context_size
    :param target_words
    :return: None
    """
    doc_indexed = []
    data_group_by_word = defaultdict(list)

    w_unknown_idx = word2idx[w_unknown]
    padding = np.empty(context_size, dtype=np.int32)
    padding.fill(word2idx[pad])

    label2sense = {}
    with open(dataset_path + '.gold.key.txt', 'r') as label_file:
        for line in label_file.readlines():
            tokens = line.rstrip().split(' ')
            label2sense[tokens[0]] = tokens[1]
    with open(dataset_path + '.data.xml', 'r') as f:
        checking_stack = []
        for line in f:
            split1 = line.split('</')
            if len(split1) == 1:
                tokens = split1[0][1:-1].split()
            elif len(split1) == 2:
                temp, text = split1.split('>')
                tokens = temp[1:].split()
            else:
                raise "Unnormal line"
            tag = tokens[0]
            if tag == "corpus":
                

    tree = html.parse(dataset_path + '.data.xml')
    for n_doc, doc in enumerate(tree.xpath('body/corpus')):
        corpus = doc.get('source')
        logger.info("Parse %s..." % corpus)
        doc_idxs = []
        position = 0
        for sent_node in doc.xpath('text/sentence'):
            for child_el in sent_node.getchildren():
                text = child_el.text
                word_idx = word2idx.get(text, w_unknown_idx)
                doc_idxs.append(word_idx)
                if child_el.tag == 'instance':
                    instance_id = child_el.get('id')
                    lemma = child_el.get('lemma')
                    pos = child_el.get('pos')
                    if lemma not in target_words:
                        target_word = TargetWord(lemma)
                        target_words[lemma] = target_word
                    target_word = target_words[lemma]
                    sense = label2sense[instance_id]
                    if sense not in target_word.label2target:
                        sense_idx = len(target_word.label2target)
                        target_word.label2target[sense] = sense_idx
                        target_word.target2label[sense_idx] = sense
                    target = target_word.label2target[sense]
                    data = Data(target_word, n_doc, position, pos, target)
                    data_group_by_word[lemma].append(data)
                position += 1

    # for n_doc, doc in enumerate(tree.getroot()):
    #     doc_idxs = []
    #     position = 0
    #     for sentence in doc:
    #         for token in sentence:
    #             text = normalize(token.text)
    #             word_idx = word2idx.get(text, w_unknown_idx)
    #             doc_idxs.append(word_idx)
    #             if token.tag == 'instance':
    #                 lemma = token.attrib['lemma']
    #                 pos = token.attrib['pos']
    #                 if lemma not in target_words:
    #                     target_word = TargetWord(lemma)
    #                     target_words[lemma] = target_word
    #                 target_word = target_words[lemma]
    #                 sense = label2sense[token.attrib['id']]
    #                 if sense not in target_word.label2target:
    #                     sense_idx = len(target_word.label2target)
    #                     target_word.label2target[sense] = sense_idx
    #                     target_word.target2label[sense_idx] = sense
    #                 target = target_word.label2target[sense]
    #                 # target = token.attrib['id']
    #                 data = Data(target_word, n_doc, position, pos, target)
    #                 data_group_by_word[lemma].append(data)
    #             position += 1
        doc_idxs = np.concatenate(
            [padding,
             np.asarray(doc_idxs, dtype=np.int32),
             padding], 0
        )
        doc_indexed.append(doc_idxs)
    return doc_indexed, data_group_by_word


def extract_words_noad(dataset_path, word2idx, context_size):
    raise NotImplementedError


def process_data(dataset_path, data_type, word2idx, target_words, context_size, augment=False):
    if data_type == "masc":
        extractor = extract_words_masc
    elif data_type == "omsti":
        extractor = extract_words_omsti
    elif data_type == "noad":
        extractor = extract_words_noad
    else:
        raise "Unknown data type"

    doc_indexed, data_group_by_word = extractor(dataset_path, word2idx, target_words, context_size)

    datasets = {}
    for key, target_word_dataset in data_group_by_word.items():
        if augment:
            # Augment data
            targets = target_words[key].target2label.keys()
            data_group_by_sense = [[data for data in target_word_dataset if data.target == target] for target in targets]
            upper_num_sense_data = max(len(data) for data in data_group_by_sense)
            for target_data in data_group_by_sense:
                target_data.extend(
                    augment_data(target_data, upper_num_sense_data - len(target_data))
                )
            datasets[key] = DataSet(concat_lists(data_group_by_sense))
        else:
            datasets[key] = DataSet(target_word_dataset)

    combined_data = concat_lists(
        [dataset.dataset for dataset in datasets.values()]
    )
    datasets[mixed_str] = DataSet(combined_data)

    return BatchConstructor(datasets, doc_indexed, context_size)
