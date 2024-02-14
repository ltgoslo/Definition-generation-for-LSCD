import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
from scipy.special import kl_div
import re

PATTERN = re.compile(r"Hva betyr \w+\?")


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


def lesk(context_sentence, ambiguous_word, pos=None, synsets=None, lang="eng"):
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
    if lang == "eng":
        context = set(context_sentence.split())
        if synsets is None:
            synsets = wordnet.synsets(ambiguous_word, lang=lang)

        if pos:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

        if not synsets:
            return None

        _, sense = max(
            (len(context.intersection(ss.definition().split())), ss) for ss in
            synsets
        )
    elif lang == "nob":
        prompt_start = re.search(PATTERN, context_sentence).span()[0]
        context_sentence = context_sentence[:prompt_start].strip()
        context = set(context_sentence)
        _, sense = max(
            (len(context.intersection(ss[1].gloss.split())), ss[1].gloss) for ss in
            synsets.iterrows()
        )

    return sense