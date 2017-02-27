# Testing Pentti Paradigm
# C.Hillar, Feb 27, 2017 (Redwood Center for Theoretical Neuroscience)

import numpy as np
from matplotlib import pyplot as plt

from pentti import Pentti


TEST = True

properties = ['Color', 'Shape', 'Currency', 'Language']
prop_to_values = {}
prop_to_values['Color'] = ['Red', 'Yellow']
prop_to_values['Shape'] = ['Round', 'Square', 'Moon']
prop_to_values['Currency'] = ['Dollar', 'Peso']
prop_to_values['Language'] = ['English', 'Spanish']

if not TEST:
    Ns = range(1, 10000, 20); trials = 200;
    SPs = [.2, .35, .4, .5]
else:
    Ns = range(1, 10000, 100); trials = 20;
    SPs = [.3, .4, .5]

###################################
# Basic version with NN denoising
###################################

P = Pentti(properties, prop_to_values)

USA = P.bundle([P.bind('Currency', 'Dollar')])
Mexico = P.bundle([P.bind('Currency', 'Peso'), P.bind('Shape', 'Square'), P.bind('Color', 'Red')])

print "The currency of USA is: ", P.word_to_char(P.query('Currency', USA))
print "The color of USA is: ", P.word_to_char(P.query('Color', USA))
print "The currency of Mexico is: ", P.word_to_char(P.query('Currency', Mexico))
print "The shape of Mexico is: ", P.word_to_char(P.query('Shape', Mexico))
print "The dollar of Mexico is: ", P.word_to_char(P.query('Dollar', P.bind(Mexico, USA)))

# Bigger test
tot1 = np.zeros((len(SPs), len(Ns))); tot2 = np.zeros((len(SPs), len(Ns)))

for sp_idx, sp in enumerate(SPs):
    for N_idx, N in enumerate(Ns):
        for t in range(trials):
            P = Pentti(properties, prop_to_values, N, sparsity=sp)
            USA = P.bundle([P.bind('Currency', 'Dollar'), P.bind('Color', 'Red'), P.bind('Shape', 'Round'), P.bind('Language', 'English')])
            Mexico = P.bundle([P.bind('Currency', 'Peso'), P.bind('Color', 'Yellow'), P.bind('Shape', 'Square'), P.bind('Language', 'Spanish')])
            if P.word_to_char(P.query('Dollar', P.bind(Mexico, USA))) == 'Peso': tot1[sp_idx, N_idx] += 1
            if P.word_to_char(P.query('Currency', USA)) == 'Dollar': tot2[sp_idx, N_idx] += 1

plt.figure(); plt.title('Nearest Neighbor Performance')
plt.ylim([0, 1])
for sp_idx, sp in enumerate(SPs):
    plt.plot(Ns, tot1[sp_idx] / trials, label='Dollar of Mexico (p=%1.3f)' % sp, linewidth=2)
    plt.plot(Ns, tot2[sp_idx] / trials, label='Currency of USA (p=%1.3f)' % sp, linewidth=2)
plt.xlabel('Dimensionality of binary vectors')
plt.ylabel('Percent correct')
plt.legend(frameon=False, loc='best')
plt.show()
