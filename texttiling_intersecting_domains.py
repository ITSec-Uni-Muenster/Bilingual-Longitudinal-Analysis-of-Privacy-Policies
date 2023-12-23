import os
import sys
import glob
import re
import datetime
from time import time
import json
import string
import warnings
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
import ujson

import scipy.sparse as ss
import numpy as np
import pandas as pd

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TextTilingTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sacremoses import MosesDetokenizer
from textacy import preprocessing as textacy_preprocessing
import pandas as pd
import ftfy
from joblib import Parallel, delayed
from tqdm import tqdm
from somajo import SoMaJo

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel, HdpModel, LsiModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import tomotopy as tp
from bertopic import BERTopic
from top2vec import Top2Vec
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from umap import UMAP

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

#################### LOAD DATA #################################

def process_texts(language):
    df_gdpr = pd.read_json("./data/gdpr_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    df_ccpa = pd.read_json("./data/ccpa_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    df_feb2021 = pd.read_json("./data/feb2021_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    df_cpra = pd.read_json("./data/cpra_" + language + "_policies_intersecting_policy_domains.json.tar.gz", orient="records", lines=True)
    list_of_textids = df_gdpr["TextID"].tolist() + df_ccpa["TextID"].tolist() + df_feb2021["TextID"].tolist() + df_cpra["TextID"].tolist()
    list_of_texts = df_gdpr["Text"].tolist() + df_ccpa["Text"].tolist() + df_feb2021["Text"].tolist() + df_cpra["Text"].tolist()
    list_of_crawls = df_gdpr["Crawl"].tolist() + df_ccpa["Crawl"].tolist() + df_feb2021["Crawl"].tolist() + df_cpra["Crawl"].tolist()
    print("Cleaning the texts ...", flush=True)
    list_of_texts = Parallel(n_jobs=-2)(delayed(text_cleaner)(text) for text in tqdm(list_of_texts, desc="Cleaning texts"))
    print("Splitting the texts ...", flush=True)
    text_splitter(list_of_texts, list_of_crawls, list_of_textids, language)

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
    # the rest was extracted from the texts and sanitized manually.
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
    text = textacy_preprocessing.replace.emails(text, "REPLACEDEMAIL")
    text = textacy_preprocessing.replace.urls(text, "REPLACEDURL")
    text = textacy_preprocessing.replace.phone_numbers(text, "REPLACEDPHONENUMBER")
    text = re.sub(" +", " ", "".join(x if x.isprintable() or x in string.whitespace else " " for x in text))
    text = text.replace("\n", "\n\n") # credit: https://stackoverflow.com/questions/69870891/nltk-texttiling
    text = replace_company_names(text)
    return text


def text_splitter(list_of_texts, list_of_crawls, list_of_textids, language):
    list_of_multiplied_crawls = []
    # list_of_lengths_of_splitted_texts = []
    splitted_by_texttile = 0
    splitted_by_splitlines = 0
    number_of_tiles = 0
    
    tt = TextTilingTokenizer(stopwords=stopwords_list_aggregation(language))
    for text, crawl, textid in zip(tqdm(list_of_texts, total=len(list_of_texts), desc="TextTiling"), list_of_crawls, list_of_textids):
        try:
            splitted_text = tt.tokenize(text)
            if len(splitted_text) == 1:
                splitted_text = text.splitlines()
                splitted_by_splitlines += 1
            else:
                splitted_by_texttile += 1
        except ValueError:
            splitted_text = text.splitlines()
            splitted_by_splitlines += 1
        splitted_text = [text.replace("\n", " ") for text in splitted_text]
        splitted_text = list(filter(None, splitted_text))
        # list_of_lengths_of_splitted_texts += [len(gensim.utils.simple_preprocess(text)) for text in splitted_text] We don't need it --> save time
        splitted_text = [text for text in splitted_text if len(gensim.utils.simple_preprocess(text))>3] # filter out paragraphs with less than 3 words
        multiplied_crawl = [crawl] * len(splitted_text)
        multiplied_textid = [textid] * len(splitted_text)
        list_of_multiplied_crawls += multiplied_crawl
        number_of_tiles += len(splitted_text)
        save_text_tiles(splitted_text, multiplied_crawl, multiplied_textid, language)
    
    assert (number_of_tiles) == (len(list_of_multiplied_crawls))
    print("Number of texts: {}".format(len(list_of_texts)))
    print("Number of texts per crawl: {}".format(dict(Counter(list_of_crawls))))
    print("Number of Tiles: {}".format(number_of_tiles))
    print("Number of Tiles per crawl: {}".format(dict(Counter(list_of_multiplied_crawls))))
    # print("Distribution of tokens per Tile: {}".format(dict(Counter(list_of_lengths_of_splitted_texts))))
    print("Distribution of splitted texts by texttiling or splitlines: {}".format({"TextTiling": splitted_by_texttile, "Splitlines": splitted_by_splitlines}))


def save_text_tiles(list_of_texts, list_of_crawls, list_of_textids, language):
    with open("./data/texttiled_cleaned_" + language + "_policies_intersecting_policy_domains.ndjson", "a") as f_writer:
        writer = ndjson.writer(f_writer, ensure_ascii=False)
        for text, crawl, textid in zip(list_of_texts, list_of_crawls, list_of_textids):
            dict_of_tiles = {"TextID":textid, "Crawl":crawl, "Text":text}
            writer.writerow(dict_of_tiles)


if __name__ == '__main__':
    print("Start time: ", str(datetime.datetime.now()), flush=True)
    t0 = time()
    print(tp.isa, flush=True)
    language_list = ["de", "en"]
    for language in tqdm(language_list, desc="language"):
        process_texts(language)

    print("End time: ", str(datetime.datetime.now()), flush=True)
    print("done in %0.3fs." % (time() - t0), flush=True)
