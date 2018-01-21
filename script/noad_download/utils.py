
lexid_map_path = '/local/scratch/xzhu/work/lexid_map.csv' # TODO


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

def map_pos(pos):
    the_map = {
        'noun':'n',
        'adjective':'a',
        'adj':'a',
        'verb':'v',
        'adv':'r',
        'adverb':'r',
        'pron':'pron',
        'adp':'adp',
        'conj':'conj',
        'det':'det',
        'num':'num',
        'prt':'prt',
        'x':'x',
        'other':'x',
        'preposition':'apd',
    }
    return the_map.get(pos.lower(), None)
