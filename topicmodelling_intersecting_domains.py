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
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from umap import UMAP

import faulthandler
faulthandler.enable()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


set_of_gdpr_crawls = {"2017-12-06",
                       "2018-01-22",
                       "2018-02-26",
                       "2018-03-26",
                       "2018-04-24",
                       "2018-05-07",
                       "2018-05-18",
                       "2018-05-25",
                       "2018-06-28",
                       "2018-07-18",
                       "2018-08-21",
                       "2018-09-21",
                       "2018-10-12",
                       "2018-11-30",
                       "2018-12-28"}

set_of_ccpa_cpra_crawls = {"2019-12-09",
                       "2020-01-02",
                       "2020-01-16",
                       "2020-02-03",
                       "2020-02-26",
                       "2020-03-09",
                       "2020-03-23",
                       "2020-04-09",
                       "2020-04-27",
                       "2020-05-13",
                       "2020-05-22",
                       "2020-06-09",
                       "2020-06-23",
                       "2020-07-09",
                       "2020-07-30",
                       "2021-02-18",
                       "2023-01-14",
                       "2023-02-13"}


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
                        'XML_entity',
                        'amount',
                        'time',
                        'abbreviation',
                        'number',
                        'number_compound',
                        'date',
                        'regular',
                        'semester',
                        'URL',
                        'email_address',
                        'symbol',
                        'emoticon',
                        'hashtag',
                        'ordinal',
                        'action_word'}

dict_of_umlaute_errors = {**dict_of_umlaute_errors,
                          **{key.lower(): value for key, value in dict_of_umlaute_errors.items()}}

#################### LOAD DATA #################################

def load_text_and_crawls(language, scenario):
    # Preprocessing, cleaning and Text tiling was already performed separately. 
    df = pd.read_json("data/texttiled_cleaned_" + language + "_policies_intersecting_policy_domains.json", orient="records", lines=True)
    if scenario == "alexa":
        df.query("Crawl in @set_of_gdpr_crawls", inplace=True)
    elif scenario == "tranco":
        df.query("Crawl in @set_of_ccpa_cpra_crawls", inplace=True)
    else:
        print("Scenario not implemented")
        sys.exit()
    passages = df["Text"].tolist()
    crawls = df["Crawl"].tolist()
    assert len(passages) == len(crawls)
    print(f"Number of passages in {language} {scenario}: {len(passages)}")
    return passages, crawls


################### DATA PROCESSING ##########################

def stopwords_list_aggregation(language):
    if language == "en":
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
    else:
        list_of_stopwords = stopwords.words(iso_639_1_codes[language])
        list_of_stopwords += [unicodedata.normalize("NFD", sw).encode("ASCII", "ignore").decode("UTF-8") for sw in list_of_stopwords]
        list_of_stopwords = sorted(list(set(sorted(list_of_stopwords))))
    return list_of_stopwords


def somajo_tokenizer(text):
    list_of_lists_of_tokens = tokenizer.tokenize_text([text])
    tokens = [token.text for list_of_tokens in list_of_lists_of_tokens for token in list_of_tokens if token.token_class in ["regular", "abbreviation", "number_compound", "number"]]
    return tokens


def berttopic_topicmodeling(docs, list_of_crawls, language, scenario):

    vectorizer = CountVectorizer(tokenizer=somajo_tokenizer, ngram_range=(1, 3), stop_words=stopwords_list_aggregation(language), min_df=20)

    # umap_model = UMAP(n_neighbors=15,
    #                  n_components=5,
    #                  min_dist=0.0,
    #                  metric='cosine',
    #                  low_memory=False,
    #                  random_state=1337)

    representation_model = MaximalMarginalRelevance(diversity=0.1)
    # topic_model = BERTopic(language="multilingual", min_topic_size=20, low_memory=True, nr_topics="auto", embedding_model="paraphrase-multilingual-MiniLM-L12-v2", umap_model=umap_model, representation_model=representation_model, vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)
    topic_model = BERTopic(language="multilingual", min_topic_size=20, low_memory=True, nr_topics="auto", embedding_model="paraphrase-multilingual-MiniLM-L12-v2", representation_model=representation_model, vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)
    topic_model.fit(docs)
    Path("./processed/topic_modeling/" + scenario + "_" + language).mkdir(parents=True, exist_ok=True)
    topic_model.save("./processed/topic_modeling/" + scenario + "_" + language, serialization="safetensors", save_ctfidf=True, save_embedding_model="paraphrase-multilingual-MiniLM-L12-v2")

    print(topic_model.get_topic_info(), flush=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("./results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topic_info.csv", index=False, sep=";", encoding="utf-8")
    del topic_info

    topics = pd.DataFrame.from_dict(topic_model.get_topics(), orient="index")
    topics.to_csv("./results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topics.csv", sep=";", encoding="utf-8") # index is necessary here.
    del topics

    docs_info = topic_model.get_document_info(docs)
    docs_info.to_csv("./results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_document_information.csv", sep=";", encoding="utf-8")
    del docs_info

    topics_over_time = topic_model.topics_over_time(docs=docs, timestamps=list_of_crawls, datetime_format="%Y-%m-%d")
    print(topics_over_time, flush=True)
    topics_over_time.to_csv("results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topics_over_time.csv", index=False, sep=";", encoding="utf-8")

    fig = topic_model.visualize_topics_over_time(topics_over_time, normalize_frequency=True)
    fig.write_html("results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topics_over_time.html")
    del fig
    fig = topic_model.visualize_topics()
    fig.write_html("results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topics.html")
    del fig
    fig = topic_model.visualize_term_rank()
    fig.write_html("results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_term_rank.html")
    del fig
    topic_distr, _ = topic_model.approximate_distribution(docs)
    np.save("./results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_topic_distributions.npy", topic_distr)
    # hierarchical_topics = topic_model.hierarchical_topics(docs)
    # hierarchical_topics.to_csv("results/topic_modeling/" + scenario + "_scenario_" + language + "_bertopic_hierarchical_topics.csv", index=False, sep=";", encoding="utf-8")
    # del hierarchical_topics


def json_saver(list_of_dicts, language, crawl_date):
    with open("results/topic_modeling/topics_of_" + crawl_date + "_" + language + "_documents.json", mode="w",
              encoding="utf-8") as f:
        json.dump(list_of_dicts, f, ensure_ascii=False)


if __name__ == '__main__':
    print("Start time: ", str(datetime.datetime.now()), flush=True)
    t0 = time()
    print(tp.isa, flush=True)
    scenario = sys.argv[1]
    language = sys.argv[2]
    list_of_texts, list_of_crawls = load_text_and_crawls(language, scenario)
    somajo_languages = {"de": "de_CMC", "en": "en_PTB"}
    tokenizer = SoMaJo(somajo_languages[language], split_sentences=False)
    berttopic_topicmodeling(list_of_texts, list_of_crawls, language, scenario)

    print("End time: ", str(datetime.datetime.now()), flush=True)
    print("done in %0.3fs." % (time() - t0))
