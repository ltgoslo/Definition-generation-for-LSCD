from time import sleep

import pandas as pd
from bs4 import BeautifulSoup
from requests import get
import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet
from scipy.special import kl_div
import re

PATTERNS = {
    "norwegian1": re.compile(r"Hva betyr \w+\?"),
    "norwegian2": re.compile(r"Hva betyr \w+\?"),
    "russian1": re.compile(r"Что такое \w+\?"),
    "russian2": re.compile(r"Что такое \w+\?"),
    "russian3": re.compile(r"Что такое \w+\?"),
}
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36',
}

def kl(sense_ids1, sense_ids2, no_zeros=False):
    sum1 = sum(sense_ids1)
    sum2 = sum(sense_ids2)
    sense_ids1 = [x / sum1 for x in sense_ids1]
    sense_ids2 = [x / sum2 for x in sense_ids2]

    if no_zeros:
        if 0 in sense_ids1:
            sense_ids1 = [float(x + 0.00001) for x in sense_ids1]
        if 0 in sense_ids2:
            sense_ids2 = [float(x + 0.00001) for x in sense_ids2]
    return sum(kl_div(sense_ids1, sense_ids2))


def lesk(
        context_sentence,
        ambiguous_word,
        pos=None,
        synsets=None,
        lang="english",
):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.

    This function is an implementation of the original Lesk algorithm (1986) [1].

    Usage example::

        >>> lesk(['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.'], 'bank', 'n')
        Synset('savings_bank.n.02')

    [1] Lesk, Michael. "Automatic sense disambiguation using machine
    readable dictionaries: how to tell a pine cone from an ice cream
    cone." Proceedings of the 5th Annual International Conference on
    Systems Documentation. ACM, 1986.
    https://dl.acm.org/citation.cfm?id=318728
    """
    if lang == "english":
        context = set(context_sentence.split())
        if synsets is None:
            synsets = wordnet.synsets(ambiguous_word)

        if pos:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

        if not synsets:
            return None

        _, sense = max(
            (len(context.intersection(ss.definition().split())), ss) for ss in
            synsets
        )
    elif ("norwegian" in lang) or ("russian" in lang):
        prompt_start = re.search(PATTERNS[lang], context_sentence).span()[0]
        context_sentence = context_sentence[:prompt_start].strip()
        context = set(context_sentence.lower().split())
        _, sense = max(
            (len(context.intersection(gloss.split())), gloss) for gloss in
            synsets.gloss.unique()
        )
    return sense


def read_html_wiktionary(word):
    source = f"https://ru.wiktionary.org/wiki/{word}"
    response = get(source, headers=HEADERS)
    contents = response.text

    soup = BeautifulSoup(contents, "html.parser")
    header_element = soup.find(attrs={'id': 'Значение'}).parent
    this_word = {"word": word, "gloss": []}
    meanings_element = header_element.next_sibling.next_sibling
    for li in meanings_element.children:
        gloss = li.text.split("◆")[0].strip()
        if gloss:
            this_word["gloss"].append(gloss)
    sleep(0.1)
    return pd.DataFrame(this_word)