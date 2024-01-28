
from nltk.corpus import wordnet as wn

def get_wordnet_name(wordnet_id):
    synset = wn.synset_from_pos_and_offset('n', int(wordnet_id[1:]))
    return synset.lemmas()[0].name()