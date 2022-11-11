import nltk


def words_of_sentences(db):
    for sentence in nltk.sent_tokenize(db):
        yield nltk.word_tokenize(sentence)