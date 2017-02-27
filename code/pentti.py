# Bare-bones Python implementation of Pentti Kanerva hyper-computing binding
# Kanerva, Pentti. "What We Mean When We Say" What's the Dollar of Mexico?":
# Prototypes and Mapping in Concept Space." AAAI Fall Symposium:
# Quantum Informatics for Cognitive, Social, and Semantic Processes. 2010.
# https://pdfs.semanticscholar.org/f477/232c0a0835dcbc4fc6b6283db484695482f9.pdf
#
# C. Hillar, Feb. 25, 2017 (Redwood Center for Theoretical Neuroscience)

import numpy as np


class Pentti(object):
    """ properties: e.g. 'Color'
        values per property: e.g. 'Yellow'

        TODO: -allow dynamic updating of new properties, etc
              -better binding and bundling schemes
    """
    def __init__(self, properties, prop_to_values, dimension=11, sparsity=.5):
        """ Basic version with Nearest Neighbor denoising """
        self.properties = properties
        self.prop_to_values = prop_to_values  # dictionary
        self.dimension = dimension
        self.sparsity = sparsity
        self.char_to_idx = {}
        self.idx_to_char = []

        c = 0
        for prop in properties:
            num = len(prop_to_values[prop])
            self.char_to_idx[prop] = c
            self.idx_to_char.append(prop)
            c += 1
            for value in prop_to_values[prop]:
                self.char_to_idx[value] = c
                self.idx_to_char.append(value)
                c += 1
        self.num_words = c

        self.all_words = self.init_words()

    def init_words(self):
        return (np.random.random((self.num_words, self.dimension)) > self.sparsity).astype(np.double)

    def char_to_word(self, label):
        return self.all_words[self.char_to_idx[label]]

    def word_to_char(self, word):
        for i in range(self.num_words):
            if (self.all_words[i] == word).all():
                return self.idx_to_char[i]

    def bind(self, property, value):
        """ override this """
        if value.__class__ == str:
            value = self.char_to_word(value)
        if property.__class__ == str:
            property = self.char_to_word(property)
        new_word = property + value
        new_word[new_word==2] *= 0
        return new_word

    def bundle(self, words):
        """ override this """
        total = np.zeros(self.dimension)
        for word in words:
            total += word
        total[total < (len(words) / 2.)] = 0
        total[total >= (len(words) / 2.)] = 1
        return total

    def process(self, word):
        """ override this """
        return word

    def clean_up(self, word):
        """ default Nearest Neighbor lookup: override this """
        word = self.process(word)
        min_dist = len(word)
        min_word = word
        for w in self.all_words:
            dist = np.abs(w - word).sum()
            if dist < min_dist:
                min_word = w
                min_dist = dist
        return min_word

    def query(self, property, word):
        return self.clean_up(self.bind(property, word))


class PenttiReal(Pentti):
    """ TODO: Simple Real valued version """
    def __init__(self, *args):
        super(PenttiMPF,self).__init__(*args)

    def init_words(self):
        return np.random.randn(self.num_words, self.dimension)

    def bind(self, property, value):
        pass

    def bundle(self, words):
        pass
