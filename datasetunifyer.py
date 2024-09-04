#%%
import sys
import os

import glob

import pandas as pd
import natsort as ns
#%%
sys.path.append(os.path.abspath(os.path.join('')))
sys.path.append(os.path.abspath(os.path.join('FakeNilc')))
sys.path.append(os.path.abspath(os.path.join('FakeNilc','fakenilc')))
sys.path.append(os.path.abspath(os.path.join('Pandas2ARFF')))
#%%
# from FakeNilc.fakenilc.preprocess import liwc, bow, pos, syntax, metrics
from FakeNilc.fakenilc.extract import loadCorpus
from Pandas2ARFF.pandas2arff import pandas2arff
#%%
def file_to_df(path):
    texts_list = []
    list_of_files = glob.glob(os.path.join(path,'*.txt'))
    for file_name in list_of_files:
        with open(file_name, 'r', encoding='utf8') as text:
            texts_list.append(text.read())
    texts_df = pd.DataFrame(texts_list, columns=['news_text_full'])
    return texts_df

def corpus_to_df(path, metadata_columns):
    def load_corpus_text(path_full, column_name):
        _, filenames, _ = loadCorpus(path_full)
        texts = []
        news = []
        unicodes_to_strip = {
            "\n\n": " ",
            "\n": " ",
            "\ufeff": "",
            "\x85": "",
            "\x91": "",
            "\x92": "",
            "\x93": "",
            "\x94": "",
            "\x96": "",
            "\x97": "",
            "\t": ""
        }
        for file_name in filenames:
            with open(file_name, 'r', encoding='utf8') as text:
                news = text.read()
                for key, value in unicodes_to_strip.items():
                    news = news.replace(key, value)
                texts.append(news)

        text_df = pd.DataFrame(texts, columns=[column_name])

        return text_df

    def load_meta(path, metadata_columns):
        meta_ids = []
        meta_filenames = []
        meta_tags = []

        for filename in os.listdir(os.path.join(path, 'full_texts',
                                                'true-meta-information')):
            meta_ids.append(filename.replace('-meta.txt', '-REAL'))
            meta_filenames.append(os.path.join(path, 'full_texts',
                                               'true-meta-information',
                                               filename))
            meta_tags.append('REAL')

        # From the fake news folder
        for filename in os.listdir(os.path.join(path, 'full_texts',
                                                'fake-meta-information')):
            meta_ids.append(filename.replace('-meta.txt', '-FAKE'))
            meta_filenames.append(os.path.join(path, 'full_texts',
                                               'fake-meta-information',
                                               filename))
            meta_tags.append('FAKE')

        meta_ids, meta_filenames, meta_tags = (list(t) for t in zip(*sorted(zip(
            meta_ids,
            meta_filenames,
            meta_tags))))

        meta_ids = pd.DataFrame(meta_ids, columns=['Id'])
        meta_tags = pd.DataFrame(meta_tags, columns=['Tag'])

        metadatas = []
        for filename in meta_filenames:
            with open(filename, 'r', encoding='utf8') as text:
                metadatas.append(text.read().splitlines())

        data_df = pd.DataFrame(metadatas, columns=metadata_columns)
        meta_df = pd.concat([meta_ids, data_df, meta_tags], axis=1)

        #print(meta_df.head())
        #print(metadata_columns)
        return meta_df

    news_text_full_df = load_corpus_text(os.path.join(path, 'full_texts'),
                                         'news_text_full')
    news_text_normalized_df = load_corpus_text(os.path.join(
                                                    path,
                                                    'size_normalized_texts'),
                                               'news_text_normalized')
    news_meta_df = load_meta(path, metadata_columns)

    result_df = pd.concat([news_text_full_df,
                           news_text_normalized_df,
                           news_meta_df], axis=1)
    #print(result_df)
    #print(ns.natsorted(result_df['Id'].unique()))

    result_df['Id'] = pd.Categorical(
                            result_df['Id'],
                            ordered=True,
                            categories=ns.natsorted(result_df['Id'].unique()))
    result_df = result_df.sort_values('Id')

    result_df = result_df.set_index('Id')

    return result_df

def corpus_df ():
    metadata_columns = [
        'author',
        'link',
        'category',
        'date_of_publication',
        'number_of_tokens',
        'number_of_words_without_punctuation',
        'number_of_types',
        'number_of_links_inside_the_news',
        'number_of_words_in_upper_case',
        'number_of_verbs',
        'number_of_subjuntive_and_imperative_verbs',
        'number_of_nouns',
        'number_of_adjectives',
        'number_of_adverbs',
        'number_of_modal_verbs_(mainly_auxiliary_verbs)',
        'number_of_singular_first_and_second_personal_pronouns',
        'number_of_plural_first_personal_pronouns',
        'number_of_pronouns',
        'pausality',
        'number_of_characters',
        'average_sentence_length',
        'average_word_length',
        'percentage_of_news_with_speeling_errors',
        'emotiveness',
        'diversity']

    corpus_df_data = corpus_to_df(os.path.join('data'), metadata_columns)
    return corpus_df_data

def corpus_arff ():
    corpus_arff_data = corpus_df().reset_index()
    corpus_arff_data.astype({'Id': 'object'}).dtypes
    corpus_arff_data.index.name = "id"
    pandas2arff(corpus_arff_data,
                os.path.join('data', 'fakeBR.arff'),
                wekaname="FakeBRCorpus",
                cleanstringdata=False,
                cleannan=False)
    return corpus_arff_data

def main():
    #print(corpus_df.head(50))
    #print(corpus_df.loc['24-FAKE', 'news_text_full'])

    # CSV dataset creation:
    corpus_df().to_csv(os.path.join('data',
                                    'real_and_fake_news_corpus_pt_br.csv'))

    # WEKA ARFF dataset creation:
    corpus_arff()


if __name__ == '__main__':
    main()
