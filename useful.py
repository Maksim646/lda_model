import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from gensim import corpora, models, similarities

import pymorphy2


uselessChars = [',', '.', '?', '!', ':', ';', '"', '-', '–', '<', '>', '»', '«', '=', '…', '/', '—']
digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

stop_words = ["мистер", "миссис", "был", "и", "в", "на", "с", "что",
     "и", "по", "с", "что", "они", "как", "на", "а", "от",
     "к", "это", "не", "для", "о", "его", "все", "так",
     "но", "или", "же", "да", "чтобы", "их", "когда",
     "если", "где", "кто", "который", "очень", "теперь",
     "еще", "только", "все", "тогда", "теперь", "уж",
     "между", "без", "снова", "лишь", "опять", "также",
     "тогда", "при", "после", "до", "как", "будто", "*", "ведь", "почему", "center"]

result_text = []


def remove_digits_from_text(text):
    for digit in digits:
        text = text.replace(digit, '')
    return text

def remove_useless_chars_from_text(text):
    for char in uselessChars:
            if char == '-':
                text = text.replace(char, ' ')
            text = text.replace(char, '')
    return text

def tokenize(texts = list[str]) -> list[list[str]]:
    tokens_list = []
    for text in texts:
        tokens = [w for w in word_tokenize(text)]
        tokens_list.append(tokens)
    return tokens_list


def lemmatize(texts: list[list[str]]) -> list[list[str]]:
    lemmatize_texts = []

    morph = pymorphy2.MorphAnalyzer()

    for text in texts:
        lemmatized_words = []
        for word in text:
            parsed_word = morph.parse(word)
            if parsed_word:
                normal_form = parsed_word[0].normal_form
                if normal_form not in stop_words:
                    lemmatized_words.append(normal_form)
        lemmatize_texts.append(lemmatized_words)

    return lemmatize_texts


def create_dictionary(texts: list[list[str]]) -> corpora.Dictionary:
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    return dictionary

def create_corpus(texts: list[list[str]], dictionary: corpora.Dictionary) -> list:
    return [dictionary.doc2bow(text) for text in texts]

def lda_model(corpus, dictionary):
    lda = models.LdaModel(corpus, num_topics=40, id2word=dictionary, passes=15)
    return lda


