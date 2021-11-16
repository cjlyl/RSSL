from cupy_util import *
import numpy as np


class WordEmbeddings(object):
    def __init__(self):
        self.num_words = 0
        self.total_count = 0
        self.words = []
        self.embedding_dim = 0
        self.vectors = np.zeros((0, 0))
        self.testVectors = np.zeros((0, 0))
        self.transformed_vectors = np.zeros((0, 0))
        self.counts = np.zeros(0, dtype=int)
        self.probs = np.zeros(0)
        self.word_dict = dict([])
        self.id2words = dict([])
        self.embs = dict([])

    def load_from_word2vec(self, dataDir,train_max,test_max):
        vec_file = dataDir #for Zhang_dataset
        vec_fs = open(vec_file,encoding='utf-8', errors='surrogateescape')
        line = vec_fs.readline()
        tokens = line.split()
        self.num_words = min(int(tokens[0]),test_max)
        self.embedding_dim = int(tokens[1])
        self.vectors = np.zeros((self.num_words, self.embedding_dim))
        self.testVectors = np.zeros((self.num_words, self.embedding_dim))
        self.counts = np.zeros(self.num_words, dtype=int)
        self.probs = np.ones(self.num_words)
        for i in range(self.num_words):
            word, vec = vec_fs.readline().split(' ', 1)
            self.words.append(word)
            self.word_dict[word] = i
            self.id2words[i] = word
            self.testVectors [i] = np.fromstring(vec, sep=' ', dtype='float32')
        self.vectors = self.testVectors
        # self.vectors = iter_norm(self.vectors,['center'])
        # self.testVectors = iter_norm(self.testVectors, ['center'])
        vec_fs.close()

    def save_transformed_vectors(self, filename):
        with open(filename, 'w') as fout:
            fout.write(str(self.num_words) + ' ' + str(self.embedding_dim) + '\n')
            for i in range(self.num_words):
                fout.write(self.words[i] + ' ' + ' '.join(str(x) for x in self.transformed_vectors[i]) + '\n')
            print('saving done!')


def load_embeddings(dataDir, train_max,test_max):
    print('Loading monolingual embeddings from', dataDir)
    we = WordEmbeddings()
    we.load_from_word2vec(dataDir, train_max=train_max,test_max=test_max)
    # we.vectors = iter_norm(we.vectors,normalize='renorm,center,renorm,center,renorm,center,renorm,center,renorm,center,renorm')
    normalize(we.vectors, actions=['unit', 'center', 'std', 'unit'])
    normalize(we.testVectors, actions=['unit', 'center', 'std', 'unit'])
    return we


def load_lexicon(file):
    induced_train = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word, trans = line.split()
            induced_train[word] = trans
    return induced_train


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix ** 2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def get_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    return avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix ** 2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def mean_std(matrix):
    xp = get_array_module(matrix)
    std = xp.std(matrix, axis=0)
    matrix /= std


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)
        elif action == 'std':
            mean_std(matrix)
