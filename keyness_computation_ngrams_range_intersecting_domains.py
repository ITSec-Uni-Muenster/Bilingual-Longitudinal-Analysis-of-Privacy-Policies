import os
import sys
import glob
import re
import datetime
import json
import csv
import pickle
import string
import warnings
import psutil
import unicodedata
import multiprocessing as mp
import traceback
import sqlite3
import os.path
import io

from pathlib import Path
from collections import Counter
from pprint import pprint
import ndjson

import scipy.sparse as ss
import numpy as np
import pandas as pd

import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from textacy import preprocessing as textacy_preprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from somajo import SoMaJo
import ftfy

from corpus_toolkit import corpus_tools as ct
from keyness import log_likelihood, type_dist, freq_dist # code was changed to output the expected values as well.

import faulthandler
faulthandler.enable()

iso_639_1_codes = {
    'af': 'afrikanns',
    'ar': 'arabic',
    'bg': 'bulgarian',
    'bn': 'bengali',
    'ca': 'catalan',
    'cs': 'czech',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'fa': 'persian',
    'fi': 'finnish',
    'fr': 'french',
    'he': 'hebrew',
    'hr': 'croatian',
    'hu': 'hungarian',
    'id': 'indonesian',
    'it': 'italian',
    'ko': 'korean',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'mk': 'macedonian',
    'ml': 'malayalam',
    'nl': 'dutch',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sk': 'slovak',
    'sl': 'slovenian',
    'sq': 'albanian',
    'sv': 'swedish',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'vi': 'vietnamese',
    'zh-cn': 'chinese'
}

spacy_languages = {"de": "de_core_news_lg",
                   "el": "el_core_news_lg",
                   "en": "en_core_web_trf",
                   "es": "es_core_news_lg",
                   "fr": "fr_core_news_lg",
                   "it": "it_core_news_lg",
                   "nl": "nl_core_news_lg",
                   "pt": "pt_core_news_lg",
                   "xx": "xx_ent_wiki_sm",
                   "nb": "nb_core_news_lg",
                   "lt": "lt_core_news_lg",
                   "zh": "zh_core_web_lg",
                   "da": "da_core_news_lg",
                   "ja": "ja_core_news_lg",
                   "pl": "pl_core_news_lg",
                   "ro": "ro_core_news_lg"}


dict_of_crawls_to_dates = {"2017-12": "2017-12-06",
                            "2018-01": "2018-01-22",
                             "2018-02": "2018-02-26",
                             "2018-03": "2018-03-26",
                             "2018-04": "2018-04-24",
                             "2018-05-0": "2018-05-07",
                             "2018-05-1": "2018-05-18",
                             "2018-05-2": "2018-05-25",
                             "2018-06": "2018-06-28",
                             "2018-07": "2018-07-18",
                             "2018-08": "2018-08-21",
                             "2018-09": "2018-09-21",
                             "2018-10": "2018-10-12",
                             "2018-11": "2018-11-30",
                             "2018-12": "2018-12-28",
                             "201912": "2019-12-09",
                             "202001-1": "2020-01-02",
                             "202001-2": "2020-01-16",
                             "202002-1": "2020-02-03",
                             "202002-2": "2020-02-26",
                             "202003-1": "2020-03-09",
                             "202003-2": "2020-03-23",
                             "202004-1": "2020-04-09",
                             "202004-2": "2020-04-27",
                             "202005-1": "2020-05-13",
                             "202005-2": "2020-05-22",
                             "202006-1": "2020-06-09",
                             "202006-2": "2020-06-23",
                             "202007-1": "2020-07-09",
                             "202007-2": "2020-07-30",
                             "2021-2": "2021-02-18",
                             "2023-1": "2023-01-14",
                             "2023-2": "2023-02-13"}


dict_of_umlaute_errors = {'Ã¼':'ü',
                            'Ã¤':'ä',
                            'Ã¶':'ö',
                            'Ã–':'Ö',
                            'ÃŸ':'ß',
                            'Ã ':'à',
                            'Ã¡':'á',
                            'Ã¢':'â',
                            'Ã£':'ã',
                            'Ã¹':'ù',
                            'Ãº':'ú',
                            'Ã»':'û',
                            'Ã™':'Ù',
                            'Ãš':'Ú',
                            'Ã›':'Û',
                            'Ãœ':'Ü',
                            'Ã²':'ò',
                            'Ã³':'ó',
                            'Ã´':'ô',
                            'Ã¨':'è',
                            'Ã©':'é',
                            'Ãª':'ê',
                            'Ã«':'ë',
                            'Ã€':'À',
                            'Ã':'Á',
                            'Ã‚':'Â',
                            'Ãƒ':'Ã',
                            'Ã„':'Ä',
                            'Ã…':'Å',
                            'Ã‡':'Ç',
                            'Ãˆ':'È',
                            'Ã‰':'É',
                            'ÃŠ':'Ê',
                            'Ã‹':'Ë',
                            'ÃŒ':'Ì',
                            'Ã':'Í',
                            'ÃŽ':'Î',
                            'Ã':'Ï',
                            'Ã‘':'Ñ',
                            'Ã’':'Ò',
                            'Ã“':'Ó',
                            'Ã”':'Ô',
                            'Ã•':'Õ',
                            'Ã˜':'Ø',
                            'Ã¥':'å',
                            'Ã¦':'æ',
                            'Ã§':'ç',
                            'Ã¬':'ì',
                            'Ã­':'í',
                            'Ã®':'î',
                            'Ã¯':'ï',
                            'Ã°':'ð',
                            'Ã±':'ñ',
                            'Ãµ':'õ',
                            'Ã¸':'ø',
                            'Ã½':'ý',
                            'Ã¿':'ÿ',
                            'â‚¬':'€'}


dict_of_umlaute_errors = {**dict_of_umlaute_errors, 
                          **{key.lower(): value for key, value in dict_of_umlaute_errors.items()}}


def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def fix_utf8_iso8859_errors(text):
    # source: https://sebastianviereck.de/mysql-php-umlaute-sonderzeichen-utf8-iso/
    for error, replacement in dict_of_umlaute_errors.items():
        text = text.replace(error, replacement)
    return text


def stopwords_list_aggregation_aggressiv_version(language):
    nlp = spacy.load(spacy_languages[language])
    if language == "en":
        list_of_stopwords = sorted(set(list(nlp.Defaults.stop_words) + list(stopwords.words(iso_639_1_codes[language]) + list(ENGLISH_STOP_WORDS))))
    else:
        list_of_stopwords = sorted(set(list(nlp.Defaults.stop_words) + list(stopwords.words(iso_639_1_codes[language]))))
        list_of_stopwords += [unicodedata.normalize("NFD", sw).encode("ASCII", "ignore").decode("UTF-8") for sw in list_of_stopwords]
        list_of_stopwords = sorted(list(set(sorted(list_of_stopwords))))
    return list_of_stopwords


def stopwords_list_aggregation(language):
    if language == "en":
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
    else:
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
        list_of_stopwords += [unicodedata.normalize("NFD", sw).encode("ASCII", "ignore").decode("UTF-8") for sw in list_of_stopwords]
        list_of_stopwords = sorted(list(set(sorted(list_of_stopwords))))
    return list_of_stopwords


def filter_stopwords(list_of_lists_of_tokens, language):
    print("Filtering stopwords and brands ... ")
    tokens_to_ignore = stopwords_list_aggregation(language)
    list_of_lists_of_tokens = [[token for token in list_of_tokens if token not in tokens_to_ignore] for list_of_tokens in list_of_lists_of_tokens]
    return list_of_lists_of_tokens


def read_brand_names():
    # partially adopted from https://github.com/anna-hope/brandnames
    list_of_brands = []
    file_paths = ["./code/resources/brandnames_gdpr_de.txt", "./code/resources/brandnames_ccpa_de.txt"]
    for file_path in file_paths:
        with io.open(file_path, "r", encoding="utf-8") as f:
            list_of_brands += f.read().split('\n')
    list_of_brands = list(set([brand.lower() for brand in list_of_brands]))
    list_of_brands = [ftfy.fix_text(brand) for brand in list_of_brands]
    list_of_brands = [fix_utf8_iso8859_errors(brand) for brand in list_of_brands]
    return list_of_brands


def replace_company_names(list_of_lists_of_tokens):
    print("Filtering brands and cleaning text... ", flush=True)
    tokens_to_replace = set(read_brand_names())
    list_of_lists_of_tokens = [[token if token not in tokens_to_replace else "companyname" for token in list_of_tokens] for list_of_tokens in list_of_lists_of_tokens]
    return list_of_lists_of_tokens


def get_textids_of_intersecting_policy_domains(corpus, language):
    df = pd.read_json("data/" + corpus + "_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    set_of_intersecting_TextIDs = set(df["TextID"].tolist())
    return set_of_intersecting_TextIDs


def load_tokenized_corpus(corpus, language):
    """
    References:
    Penn Treebank: https://www.sketchengine.eu/penn-treebank-tagset/
    Emperist Tagset: https://sites.google.com/site/empirist2015/home/annotation-guidelines
    Tiger Tagset: https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger
    """
    classset_to_keep = {"abbreviation", "number_compound", "regular", "number"}

    if language == "de":
        tagset_to_remove = {"XY", "$,", "$(", "$.", "ART"}
    elif language == "en":
        tagset_to_remove = {":", ".", ")", "(", "$", "#", "UH", "SYM", "SENT", "LS", "DT", ",", "CC", "--"}

    list_of_textids = []
    list_of_crawls = []
    list_of_lists_of_tokens = []

    set_of_textids_of_intersecting_policy_domains = get_textids_of_intersecting_policy_domains(corpus, language)

    print("Loading tokenized text of {} {}".format(corpus, language), flush=True)
    with open("data/" + corpus + "_TextID_Crawl_TokenClassTag_" + language + "_somajo.ndjson", "r") as f:
        reader = ndjson.reader(f)
        print("Working on " + corpus + " " + language, flush=True)
        for post in tqdm(reader, desc="loading " + language + " corpus"):
            if post["TextID"] in set_of_textids_of_intersecting_policy_domains: # only intersecting policy domains
                if post["TextID"] not in list_of_textids: # filter duplicated policies if any
                    list_of_textids.append(post["TextID"])
                    list_of_crawls.append(post["Crawl"])
                    list_of_tokens = post["TokenClassTag"]
                    list_of_tokens = [token[0].lower() for token in list_of_tokens if token[1] in classset_to_keep and token[2] not in tagset_to_remove]
                    list_of_tokens = [fix_utf8_iso8859_errors(ftfy.fix_text(token)) for token in list_of_tokens]
                    list_of_tokens = [remove_punctuation(token) for token in list_of_tokens]
                    list_of_lists_of_tokens.append(list_of_tokens)

    assert len(list_of_textids) == len(set(list_of_textids)) # no duplicates
    assert len(list_of_textids) == len(list_of_crawls) == len(list_of_lists_of_tokens) 
    if corpus != "cpra":
        list_of_crawls = [dict_of_crawls_to_dates[item] for item in list_of_crawls]
    assert len(list_of_crawls) > 0
    assert len(list_of_textids) > 0
    assert len(list_of_lists_of_tokens) > 0
    print("Crawls:", set(list_of_crawls))
    print("Number of documents: {}".format(len(list_of_lists_of_tokens)), flush=True)
    return list_of_textids, list_of_crawls, list_of_lists_of_tokens


def ngrammer(list_of_tokens, n):
    n_grams_list = []
    n_grams_listed = [list_of_tokens[i:i + n] for i in range(len(list_of_tokens) - (n - 1))]
    # n-grams
    for n_gram in n_grams_listed:
        n_grams_list.append('__'.join(n_gram))
    return n_grams_list


def preprocess_ngrams(list_of_lists_of_tokens, corpus, language, n_min, n_max):
    print("Preprocessing {}_{} ngrams ...".format(corpus, language), flush=True)
    list_of_lists_of_tokens = replace_company_names(list_of_lists_of_tokens)
    for n in range(n_min, n_max+1):
        if n == 1:
            list_of_lists_of_ngrams = filter_stopwords(list_of_lists_of_tokens, language)
        elif n > 1:
            list_of_lists_of_new_ngrams = [ngrammer(list_of_tokens, n) for list_of_tokens in list_of_lists_of_tokens]
            # add up the extracted n-grams for each document
            list_of_lists_of_ngrams = [list_of_ngrams + list_of_new_ngrams for list_of_ngrams, list_of_new_ngrams in zip(list_of_lists_of_ngrams, list_of_lists_of_new_ngrams)]
    return list_of_lists_of_ngrams


def ngram_frequency(corpus, language, list_of_lists_of_ngrams, n_min, n_max):
    """
    Overall frequencies and types in each corpus for each language
    """
    freq = ct.frequency(list_of_lists_of_ngrams, calc = 'freq', normed=True) # normed over all ngrams in the dictionary
    print(ct.head(freq, hits=len(freq), filename="results/lexical_analysis/freq_dist_" + str(n_min) + "to" + str(n_max) + "_ngrams_" + corpus + "_" + language + "_intersecting_domains.tsv", sep="\t"), flush=True)
    print("RAM memory % used:", psutil.virtual_memory()[2], flush=True)
    del freq
    type = ct.frequency(list_of_lists_of_ngrams, calc = 'range', normed=True) # range for types, normed over number of documents (policies)
    print(ct.head(type, hits=len(type), filename="results/lexical_analysis/type_dist_" + str(n_min) + "to" + str(n_max) + "_ngrams_" + corpus + "_" + language + "_intersecting_domains.tsv", sep="\t"), flush=True)
    print("RAM memory % used:", psutil.virtual_memory()[2], flush=True)


def keyness_computation_inter_corpora(reference_corpus_name, corpus_name, language, reference_corpus_ngrammized, corpus_ngrammized, list_of_reference_crawls, n_min, n_max):
    # References:
    # 1. https://ucrel.lancs.ac.uk/llwizard.html 
    # 2. https://ucrel.lancs.ac.uk/people/paul/publications/rg_acl2000.pdf
    # 3. https://alvinntnu.github.io/NTNU_ENC2036_LECTURES/keyword-analysis.html
    # 4. http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/
    # 5. https://tm4ss.github.io/docs/Tutorial_4_Term_extraction.html#3_Log_likelihood
    new_reference_corpus_ngrammized = []
    for list_of_ngrams, crawl in zip(reference_corpus_ngrammized, list_of_reference_crawls):
        if reference_corpus_name == "gdpr":
            if crawl in ["2017-12-06", "2018-01-22", "2018-02-26", "2018-03-26", "2018-04-24", "2018-05-07", "2018-05-18"]:
                new_reference_corpus_ngrammized.append(list_of_ngrams)
        elif reference_corpus_name == "ccpa":
            if crawl in ["2019-12-09", "2020-01-02", "2020-01-16", "2020-02-03", "2020-02-26", "2020-03-09", "2020-03-23", "2020-04-09", "2020-04-27", "2020-05-13", "2020-05-22", "2020-06-09", "2020-06-23"]:
                new_reference_corpus_ngrammized.append(list_of_ngrams)
    del reference_corpus_ngrammized

    print('RAM memory % used:', psutil.virtual_memory()[2], flush=True)
    log_likelihood(corpus_ngrammized, new_reference_corpus_ngrammized, save_as="results/lexical_analysis/keyness_" + str(n_min) + "to" + str(n_max) + "_ngrams_G2-freq-dist_" + corpus_name + "_vs_" + reference_corpus_name + "_" + language + "_intersecting_domains.tsv", dist_func=freq_dist)
    log_likelihood(corpus_ngrammized, new_reference_corpus_ngrammized, save_as="results/lexical_analysis/keyness_" + str(n_min) + "to" + str(n_max) + "_ngrams_G2-type-dist_" + corpus_name + "_vs_" + reference_corpus_name + "_" + language + "_intersecting_domains.tsv", dist_func=type_dist)


def keyness_computation_intra_corpora(corpus_name, language, list_of_crawls, list_of_lists_of_ngrams, n_min, n_max):
    pre_enforcement_corpus = []
    post_enforcement_corpus = []

    ngram_frequency(corpus_name, language, list_of_lists_of_ngrams, n_min, n_max)
    
    for list_of_ngrams, crawl in zip(list_of_lists_of_ngrams, list_of_crawls):
        if corpus_name == "gdpr":
            if crawl in ["2017-12-06", "2018-01-22", "2018-02-26", "2018-03-26", "2018-04-24", "2018-05-07", "2018-05-18"]:
                pre_enforcement_corpus.append(list_of_ngrams)
            elif crawl in ["2018-05-25", "2018-06-28", "2018-07-18", "2018-08-21", "2018-09-21", "2018-10-12", "2018-11-30", "2018-12-28"]:
                post_enforcement_corpus.append(list_of_ngrams)
        elif corpus_name == "ccpa":
            if crawl in ["2019-12-09", "2020-01-02", "2020-01-16", "2020-02-03", "2020-02-26", "2020-03-09", "2020-03-23", "2020-04-09", "2020-04-27", "2020-05-13", "2020-05-22", "2020-06-09", "2020-06-23"]:
                pre_enforcement_corpus.append(list_of_ngrams)
            elif crawl in ["2020-07-09", "2020-07-30"]:
                post_enforcement_corpus.append(list_of_ngrams)
    
    del list_of_lists_of_ngrams, list_of_crawls

    log_likelihood(post_enforcement_corpus, pre_enforcement_corpus, save_as="results/lexical_analysis/keyness_" + str(n_min) + "to" + str(n_max) + "_ngrams_G2-freq-dist_" + corpus_name + "_post_vs_pre_" + language + "_intersecting_domains.tsv", dist_func=freq_dist)
    log_likelihood(post_enforcement_corpus, pre_enforcement_corpus, save_as="results/lexical_analysis/keyness_" + str(n_min) + "to" + str(n_max) + "_ngrams_G2-type-dist_" + corpus_name + "_post_vs_pre_" + language + "_intersecting_domains.tsv", dist_func=type_dist)


def type_frequency_per_crawl(language, list_of_lists_of_ngrams, list_of_crawls, n_min, n_max):
    list_of_dfs = []
    for crawl in dict_of_crawls_to_dates.values():
        crawl_list_of_ngrams = []
        for list_of_ngrams, ngram_crawl in zip(list_of_lists_of_ngrams, list_of_crawls):
            if crawl == ngram_crawl:
                crawl_list_of_ngrams.append(list_of_ngrams)
        type_dict = ct.frequency(crawl_list_of_ngrams, calc = 'range', normed=True) # range for types, normed over number of documents (policies)
        df = pd.DataFrame(type_dict, index=[crawl])
        list_of_dfs.append(df)
    df = pd.concat(list_of_dfs, axis=0)
    df.to_csv("results/lexical_analysis/type_dist_of_crawls_" + str(n_min) + "to" + str(n_max) + "_ngrams_" + language + "_intersecting_domains.tsv", sep="\t")


if __name__ == '__main__':
    print("Start time: ", str(datetime.datetime.now()), flush=True)

    try:
        min_n = int(sys.argv[1])
        max_n = int(sys.argv[2])
    except:
        print("Enter 'n_min' and 'n_max' for ngrams as the first and second arguments")
        sys.exit()

    _, list_of_gdpr_de_crawls, gdpr_de_corpus_tokenized = load_tokenized_corpus("gdpr", "de")
    _, list_of_ccpa_de_crawls, ccpa_de_corpus_tokenized = load_tokenized_corpus("ccpa", "de")
    _, list_of_feb2021_de_crawls, feb2021_de_corpus_tokenized = load_tokenized_corpus("feb2021", "de")
    _, list_of_cpra_de_crawls, cpra_de_corpus_tokenized = load_tokenized_corpus("cpra", "de")

    _, list_of_gdpr_en_crawls, gdpr_en_corpus_tokenized = load_tokenized_corpus("gdpr", "en")
    _, list_of_ccpa_en_crawls, ccpa_en_corpus_tokenized = load_tokenized_corpus("ccpa", "en")
    _, list_of_feb2021_en_crawls, feb2021_en_corpus_tokenized = load_tokenized_corpus("feb2021", "en")
    _, list_of_cpra_en_crawls, cpra_en_corpus_tokenized = load_tokenized_corpus("cpra", "en")

    print('RAM memory % used:', psutil.virtual_memory()[2], flush=True)
    gdpr_de_corpus_ngrammized = preprocess_ngrams(gdpr_de_corpus_tokenized, "gdpr", "de", min_n, max_n)
    del gdpr_de_corpus_tokenized
    ccpa_de_corpus_ngrammized = preprocess_ngrams(ccpa_de_corpus_tokenized, "ccpa", "de", min_n, max_n)
    del ccpa_de_corpus_tokenized
    feb2021_de_corpus_ngrammized = preprocess_ngrams(feb2021_de_corpus_tokenized, "feb2021", "de", min_n, max_n)
    del feb2021_de_corpus_tokenized
    cpra_de_corpus_ngrammized = preprocess_ngrams(cpra_de_corpus_tokenized, "cpra", "de", min_n, max_n)
    del cpra_de_corpus_tokenized

    gdpr_en_corpus_ngrammized = preprocess_ngrams(gdpr_en_corpus_tokenized, "gdpr", "en", min_n, max_n)
    del gdpr_en_corpus_tokenized
    ccpa_en_corpus_ngrammized = preprocess_ngrams(ccpa_en_corpus_tokenized, "ccpa", "en", min_n, max_n)
    del ccpa_en_corpus_tokenized
    feb2021_en_corpus_ngrammized = preprocess_ngrams(feb2021_en_corpus_tokenized, "feb2021", "en", min_n, max_n)
    del feb2021_en_corpus_tokenized
    cpra_en_corpus_ngrammized = preprocess_ngrams(cpra_en_corpus_tokenized, "cpra", "en", min_n, max_n)
    del cpra_en_corpus_tokenized


    print('RAM memory % used:', psutil.virtual_memory()[2], flush=True)
    print("type frequency per crawl for de", flush=True)
    corpus_de_ngrammized = gdpr_de_corpus_ngrammized + ccpa_de_corpus_ngrammized + feb2021_de_corpus_ngrammized + cpra_de_corpus_ngrammized
    crawls_de = list_of_gdpr_de_crawls + list_of_ccpa_de_crawls + list_of_feb2021_de_crawls + list_of_cpra_de_crawls
    type_frequency_per_crawl("de", corpus_de_ngrammized, crawls_de, min_n, max_n)
    del corpus_de_ngrammized, crawls_de

    print("type frequency per crawl for en", flush=True)
    corpus_en_ngrammized = gdpr_en_corpus_ngrammized + ccpa_en_corpus_ngrammized + feb2021_en_corpus_ngrammized + cpra_en_corpus_ngrammized
    crawls_en = list_of_gdpr_en_crawls + list_of_ccpa_en_crawls + list_of_feb2021_en_crawls + list_of_cpra_en_crawls
    type_frequency_per_crawl("en", corpus_en_ngrammized, crawls_en, min_n, max_n)
    del corpus_en_ngrammized, crawls_en

    print("keyness_computation_intra_corpora", flush=True)
    keyness_computation_intra_corpora("gdpr", "de", list_of_gdpr_de_crawls, gdpr_de_corpus_ngrammized, min_n, max_n)
    keyness_computation_intra_corpora("ccpa", "de", list_of_ccpa_de_crawls, ccpa_de_corpus_ngrammized, min_n, max_n)
    keyness_computation_intra_corpora("gdpr", "en", list_of_gdpr_en_crawls, gdpr_en_corpus_ngrammized, min_n, max_n)
    keyness_computation_intra_corpora("ccpa", "en", list_of_ccpa_en_crawls, ccpa_en_corpus_ngrammized, min_n, max_n)

    print("keyness_computation_inter_corpora", flush=True)
    keyness_computation_inter_corpora("ccpa", "feb2021", "de", ccpa_de_corpus_ngrammized, feb2021_de_corpus_ngrammized, list_of_ccpa_de_crawls, min_n, max_n)
    keyness_computation_inter_corpora("ccpa", "feb2021", "en", ccpa_en_corpus_ngrammized, feb2021_en_corpus_ngrammized, list_of_ccpa_en_crawls, min_n, max_n)
    keyness_computation_inter_corpora("ccpa", "cpra", "de", ccpa_de_corpus_ngrammized, cpra_de_corpus_ngrammized, list_of_ccpa_de_crawls, min_n, max_n)
    keyness_computation_inter_corpora("ccpa", "cpra", "en", ccpa_en_corpus_ngrammized, cpra_en_corpus_ngrammized, list_of_ccpa_en_crawls, min_n, max_n)

    print(str(datetime.datetime.now()), flush=True)