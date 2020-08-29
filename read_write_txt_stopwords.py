import os
import nltk
import natsort as ns

def import_stopwords(language):
    return stopwords_nltk = nltk.corpus.stopwords.words(language)

def save_files_stopwords(name)
    return pass

def main():
    stopwards_nltk = import_stopwords('portuguese')

    with open(os.path.join('FakeNilc', 'fakenilc', 'var', 'stopwords-pt.txt'), 'r') as words:
        stopwords = words.readlines()
    stopwords_extended = [word.rstrip('\n') for word in stopwords]

    with open(os.path.join('data', 'stopwords_nltk.txt'), 'w') as new_file:
        for line in stopwords_nltk: new_file.write(line + '\n')

    with open(os.path.join('data', 'stopwords_nltk_ordered.txt'), 'w') as new_file:
        for line in stopwords_nltk: new_file.write(line + '\n')