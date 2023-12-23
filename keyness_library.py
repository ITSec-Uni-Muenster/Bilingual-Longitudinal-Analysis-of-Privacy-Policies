"""
This module compares two lists of tokenized texts (i.e. a corpus and a reference corpus, 
calculating the log likelihood value of every token.
The code is based on the keyness Python library (https://github.com/mikesuhan/keyness) and was modified by us. See the README file for more information.
"""

import math

def type_dist(corpus):
    """Counts number of types in a corpus"""
    output = {}

    for text in corpus:
        for word_type in set(text):
            output[word_type] = output.get(word_type, 0) + 1

    return output


def freq_dist(corpus):
    """Counts number of tokens in a corpus"""
    output = {}

    for text in corpus:
        for word in text:
            output[word] = output.get(word, 0) + 1

    return output


def log_likelihood(corpus, reference_corpus, save_as=False, dist_func=freq_dist, norm_to=1000,
                   dummy_zero=.00000000000000000001, encoding='utf8', delimiter='\t'):
    """Rank orders the log likelihood values of every token in a corpus

    Arguments:
    corpus: the corpus on which log likelihood is calculated - must be an iterable of iterables (e.g. a list of tokenized texts)
    reference_corpus: what the corpus is compared against - must be an iterable of iterables (e.g. a list of tokenized texts)

    Keyword Arguments:
    save_as: saves the results as a tsv if this value is a string, returns the output as a list if it is 0, None, or False
    dist_func: the function used to count corpus and reference_corpus
    norm_to: the rate the counts are normalized to in the output
    dummy_zero: the value 0 is substituted with in order to calculate LL of words that occur in one corpus but not the other.
    setting dummy_zero to 0, None, or False will exclude words that do not occur in either corpus
    encoding: encoding used if saved as a tsv
    delimiter: delimiter used if saved as a tsv
    """
    output = []

    corpus_dist = dist_func(corpus)
    ref_corpus_dist = dist_func(reference_corpus)

    if dummy_zero:
        all_words = set(list(corpus_dist.keys()) + list(ref_corpus_dist.keys()))
    else:
        # Excludes words that do not occur in the reference corpus
        all_words = [key for key in corpus_dist.keys() if key in ref_corpus_dist.keys()]

    """
    Calculates log likelihood value (G2) of every word in the corpus.

                                    Corpus      Ref Corpus      Total
        Frequency of word           a           b               a+b
        Frequency of other words    c-a         d-b             c+d-a-b
        Total                       c           d               c+d

    Source: http://ucrel.lancs.ac.uk/llwizard.html
          https://github.com/amacanovic/KeynessMeasures/blob/master/R/measure_calculators.R
          https://lexically.net/downloads/version_64_8/HTML/formulae.html
    """

    c = sum(corpus_dist[key] for key in corpus_dist)
    d = sum(ref_corpus_dist[key] for key in ref_corpus_dist)

    def odds_ratio(a, b, c, d):
        if (d - b == 0) or (c - a == 0):
            return a / b
        else:
            return (a / (c - a)) / (b / (d - b))

    def log_ratio(a, b, c, d):
        if a == 0:
            a_norm = (0.5 / c) * 1000000
        else:
            a_norm = (a / c) * 1000000
        if b == 0:
            b_norm = (0.5 / c) * 1000000
        else:
            b_norm = (b / d) * 1000000
        return math.log2(a_norm/b_norm)

    def G2_effect_size(G2, E1, E2, c, d):
        min_E = min(E1, E2)
        if min_E == 0:
            min_E = dummy_zero
        return G2 / ((c + d) * math.log(min_E))

    def over_under_use(a, b, c, d):
        a_norm = (a / c) * 1000000
        b_norm = (b / d) * 1000000
        if (a_norm == b_norm):
            return 'equal'
        elif (a_norm > b_norm):
            return 'overuse' #
        elif (a_norm < b_norm):
            return 'underuse'

    def relative_risk(a, b, c, d):
        a_norm = (a / c) * 1000000
        b_norm = (b / d) * 1000000
        if a_norm == 0:
            a_norm =  0.5
        if b_norm == 0:
            b_norm = 0.5
        return (a_norm / b_norm)

    def percent_diff(a, b, c, d):
        a_norm = (a / c) * 1000000
        b_norm = (b / d) * 1000000
        if b_norm == 0:
            b_norm = dummy_zero
        return ((a_norm - b_norm) * 100) / b_norm

    def diff_coefficient(a, b, c, d):
        a_norm = (a / c) * 1000000
        b_norm = (b / d) * 1000000
        return (a_norm - b_norm)/(a_norm + b_norm)

    for word in all_words:
        a = corpus_dist.get(word, dummy_zero)
        b = ref_corpus_dist.get(word, dummy_zero)
        E1 = c * (a + b) / (c + d) 
        E2 = d * (a + b) / (c + d)
        G2 = 2 * ((a * math.log(a / E1)) + (b * math.log(b / E2))) # Log likelihood
        OR = odds_ratio(a, b, c, d) # odds ratio
        LR = log_ratio(a, b, c, d) # log ratio
        BIC = G2 - (math.log(c + d)) # Bayesian Information Criterion
        ELL = G2_effect_size(G2, E1, E2, c, d) # Effect size for Log Likelihood 
        overUnder = over_under_use(a, b, c, d) # Overuse or Underuse
        RR = relative_risk(a, b, c, d) # Relative Risk
        percentDiff = percent_diff(a, b, c, d) # percentage change
        diffCoef = diff_coefficient(a, b, c, d) # Difference coefficient
        NCC = (a / c) * 1000000 # Normalized frequency per million in the target corpus
        NRCC = (b / d) * 1000000 # Normalized frequency per million in the reference corpus
        # round() is used on frequencies so that dummy_zero values are rounded to 0
        output_row = word, round(G2, 3), round(BIC, 3), round(ELL, 3), round(OR, 3), round(LR, 3), round(RR, 3), round(percentDiff, 3), round(diffCoef, 3), overUnder, round(a), round(b), round(E1, 3), round(E2, 3), round(c), round(d), round(NCC, 3), round(NRCC, 3)
        output.append(output_row)

    if save_as:
        tsv(sorted(output, key=lambda x: x[1], reverse=True), save_as, encoding=encoding, delimiter=delimiter)
    else:
        return sorted(output, key=lambda x: x[1], reverse=True)


def tsv(output, save_as, encoding='utf8', delimiter='\t'):
    """Saves the result of log_likelihood as a tsv."""
    tsv = '\n'.join(delimiter.join(str(item) for item in line) for line in output)
    header = 'Word', 'LL', 'BIC', 'ELL', 'OR', 'LR', 'RR', 'PercDiff', 'DiffC', 'WordUse', 'CC', 'RCC', 'CC.E', 'RCC.E', 'CT', 'RCT', 'NCC', 'NRCC'
    tsv = delimiter.join(header) + '\n' + tsv

    with open(save_as, 'w', encoding=encoding) as f:
        f.write(tsv)

    print('saved as:', save_as)
