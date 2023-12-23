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
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from somajo import SoMaJo
import ftfy
import ndjson

from tinydb import TinyDB
from tinydb import where as tinydb_where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from corpus_toolkit import corpus_tools as ct
from keyness import log_likelihood, type_dist, freq_dist

import faulthandler
faulthandler.enable()

import faulthandler
faulthandler.enable()

lemmatization_lists = [
    'ast', 'bg', 'ca', 'cs', 'cy', 'de', 'en', 'es', 'et', 'fa', 'fr', 'ga',
    'gd', 'gl', 'gv', 'hu', 'it', 'pt', 'ro', 'sk', 'sl', 'sv', 'uk']
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
                   "en": "en_core_web_lg",
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
                             "2021-2": "2021-02-18"}


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

somaja_token_classes = {'measurement',
                        'mention',
                        'XML_tag',
                        'amount',
                        'time',
                        'abbreviation',
                        'number',
                        'number_compound',
                        'date',
                        'regular',
                        'URL',
                        'email_address',
                        'symbol',
                        'emoticon',
                        'hashtag',
                        'ordinal'}

dict_of_umlaute_errors = {**dict_of_umlaute_errors, 
                          **{key.lower(): value for key, value in dict_of_umlaute_errors.items()}}


#################### PREPROCESSING #################################

def fix_utf8_iso8859_errors(text):
    # source: https://sebastianviereck.de/mysql-php-umlaute-sonderzeichen-utf8-iso/
    for error, replacement in dict_of_umlaute_errors.items():
        text = text.replace(error, replacement)
    return text


def stopwords_list_aggregation(language):
    if language == "en":
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
    else:
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
        list_of_stopwords += [unicodedata.normalize("NFD", sw).encode("ASCII", "ignore").decode("UTF-8") for sw in list_of_stopwords]
        list_of_stopwords = sorted(list(set(sorted(list_of_stopwords))))
    return list_of_stopwords


def filter_remaining_stopwords(list_of_list_of_n_grams, language):
    print("Filtering remaining stopwords ... ")
    set_of_stopwords = set(stopwords_list_aggregation(language))
    list_of_list_of_n_grams = [[n_gram for n_gram in list_of_n_grams if n_gram not in set_of_stopwords] for list_of_n_grams in list_of_list_of_n_grams]
    return list_of_list_of_n_grams


def read_company_names():
    # Partially adopted from https://github.com/anna-hope/brandnames
    # The rest was extracted from the texts and sanitized manually.
    list_of_brands = []
    file_path = "code/resources/brandnames.txt"
    with io.open(file_path, "r", encoding="utf-8") as f:
        list_of_brands = f.read().split('\n')
    list_of_brands = list(set([brand.strip() for brand in list_of_brands]))
    list_of_brands = [ftfy.fix_text(brand) for brand in list_of_brands]
    list_of_brands = [fix_utf8_iso8859_errors(brand) for brand in list_of_brands]
    return list_of_brands


def replace_company_names(text):
    list_of_company_names = read_company_names()
    for company_name in list_of_company_names:
        try:
            # https://stackoverflow.com/questions/29996079/match-a-whole-word-in-a-string-using-dynamic-regex
            if company_name in text:
                text = re.sub(r"\b%s\b" % re.escape(company_name), "COMPANYNAME", text)
        except:
            continue
    return text


def text_cleaner(text):
    text = textacy_preprocessing.normalize.bullet_points(text)
    text = textacy_preprocessing.normalize.unicode(text)
    text = ftfy.fix_text(text)
    text = fix_utf8_iso8859_errors(text)
    text = textacy_preprocessing.normalize.hyphenated_words(text)
    text = textacy_preprocessing.normalize.whitespace(text)
    text = re.sub(" +", " ", "".join(x if x.isprintable() or x in string.whitespace else " " for x in text))
    text = replace_company_names(text)
    return text


def load_text_of_policies(corpus, language):
    df = pd.read_json("data/" + corpus + "_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    df["Text"] = Parallel(n_jobs=-1)(delayed(text_cleaner)(text) for text in tqdm(df["Text"].tolist(), desc="Cleaning texts"))
    return df


def calculate_dependency_bigrams_per_crawl(df, language, corpus):
    print("Extracting depndency bi-grams from ", corpus, flush=True)
    set_of_crawls = set(df["Crawl"].tolist())
    for crawl in set_of_crawls:
        df_crawl = df.loc[df["Crawl"] == crawl]
        bg_dict = ct.dep_bigram(df_crawl["Text"].tolist(), "dobj")
        # keys of bg_dict include "bi_freq", "dep_freq", "head_freq", and "range"
        with open("results/lexical_analysis/dependency_bigrams_" + corpus + "_" + language + "_" + str(crawl) + "_intersecting_domains.json", mode="w", encoding="utf-8") as f:
            json.dump(bg_dict, f, ensure_ascii=False)
        with open("results/lexical_analysis/dependency_bigrams_" + corpus + "_" + language + "_" + str(crawl) + "_intersecting_domains.json", mode="r", encoding="utf-8") as f:
            bg_dict = json.load(f)
            ct.dep_conc(bg_dict["samples"], "results/lexical_analysis/dependency_bigrams_concordance_lines_dobj_" + corpus + "_" + language + "_" + str(crawl) + "_intersecting_domains")


def calculate_dependency_nounchunks_per_crawl(df, language, corpus):
    print("Extracting dependency noun chuncks from ", corpus, flush=True)
    set_of_crawls = set(df["Crawl"].tolist())
    for crawl in set_of_crawls:
        print("Working on", crawl, flush=True)
        df_crawl = df.loc[df["Crawl"] == crawl]
        bg_dict = ct.nounchunk_dep(df_crawl["Text"].tolist(), "dobj")
        # keys of bg_dict include "bi_freq", "dep_freq", "head_freq", and "range"
        with open("results/lexical_analysis/dependency_nounchunks_" + corpus + "_" + language + "_" + str(crawl) + "_intersecting_domains.json", mode="w", encoding="utf-8") as f:
            json.dump(bg_dict, f, ensure_ascii=False)


def main():
    print("Start time: ", str(datetime.datetime.now()))
    list_of_corpora = ["feb2021", "gdpr", "ccpa", "cpra"]

    language = "en" # corpus_toolkit only supports English
    for corpus in list_of_corpora:
        print("Working on", corpus, language, flush=True)
        df = load_text_of_policies(corpus, language)
        calculate_dependency_bigrams_per_crawl(df, language, corpus)
        calculate_dependency_nounchunks_per_crawl(df, language, corpus)
    print("End time: ", str(datetime.datetime.now()))

if __name__ == "__main__":
    main()