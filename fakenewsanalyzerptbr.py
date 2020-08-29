#%%
import pandas as pd
import os
import numpy as np
import random

import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk import tokenize

from string import punctuation
import unidecode

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
import matplotlib.pyplot as plt
import seaborn as sns

def text_classifier(dataframe, text_column, classification_column):
    print("====================================================================")
    print("============================== START ===============================\n")
    print("LOGISTIC REGRESSION WITH BAG OF WORDS FOR THE 50 MOST FREQUENT WORDS\n")
    
    vetorize = CountVectorizer(lowercase = False,
                                max_features = 50)
    bag_of_words = vetorize.fit_transform(dataframe[text_column])
    
    print("The bag of words creatad has {} instances with {} words.\n".format(bag_of_words.shape[0],bag_of_words.shape[1]))
    
    train, test, train_class, test_class = train_test_split(bag_of_words,
                                                                dataframe[classification_column],
                                                                random_state = 42)
    
    print("Will be used {} instances for training.\n".format(train.shape[0]))
    print("Will be used {} instances for testing the trained model.\n".format(test.shape[0]))
    sparse_matrix = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
                                                       columns = vetorize.get_feature_names())
    print(sparse_matrix)

    logistic_regression = LogisticRegression(solver = 'lbfgs')
    logistic_regression.fit(train, train_class)
    accuracy = logistic_regression.score(test, test_class)
    print("=============================== END ================================")
    print("====================================================================\n")
    return accuracy

def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

def sky_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(190, 40%%, %d%%)" % random.randint(30, 60)

def real_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(130, 40%%, %d%%)" % random.randint(30, 60)

def fake_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 80%%, %d%%)" % random.randint(30, 60)

def word_cloud_complete(dataframe, text_column, classification_column):
    #%matplotlib inline
    all_words = ' '.join([text for text in dataframe[text_column]])
    #print(all_words[:3])
    print("   * Were computed a total of {} words from dataset.\n".format(len(all_words)))
    
    #infinity_mask = np.array(Image.open(os.path.join("data", "img", "infinity_solid_mask.png")))
    #cloud_mask = np.array(Image.open(os.path.join("data", "img", "cloud_mask.png")))
    mapa_brasil_mask = np.array(Image.open(os.path.join("data", "img", "mapa_brasil_mask.png")))

    cloud_of_words = WordCloud(width = 1080,
                              height = 1080,
                              max_font_size = 110,
                              collocations = False,
                              mask = mapa_brasil_mask,
                              background_color = "white",
                              contour_width = 3,
                              contour_color = (219, 236, 240)).generate(all_words)
    cloud_of_words.recolor(color_func=sky_color_func, random_state=3)
    plt.figure(figsize = (12, 10))
    plt.imshow(cloud_of_words, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    return cloud_of_words, all_words, len(all_words)

def word_cloud_fake(dataframe, text_column, classification_column):
    news = dataframe.query(classification_column + " == 0")
    all_words = ' '.join([text for text in news[text_column]])
    print("   * Were computed a total of {} words from dataset.\n".format(len(all_words)))
    #thumbs_down_mask = np.array(Image.open(os.path.join("data", "img", "thumbs_down_mask.png")))
    mirrored_thumbs_down_mask = np.array(Image.open(os.path.join("data", "img", "thumbs_down_mask_3.png")))

    cloud_of_words = WordCloud(width=1080,
                               height = 1080,
                               max_font_size = 110,
                               background_color="white",
                               collocations = False,
                               mask = mirrored_thumbs_down_mask,
                               contour_width = 3,
                               contour_color = (250, 209, 209)).generate(all_words)

    cloud_of_words.recolor(color_func=fake_color_func, random_state=3)
    plt.figure(figsize = (12,10))
    plt.imshow(cloud_of_words, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    return cloud_of_words, all_words, len(all_words)

def word_cloud_real(dataframe, text_column, classification_column):
    news = dataframe.query(classification_column + " == 1")
    all_words = ' '.join([text for text in news[text_column]])    
    print("   * Were computed a total of {} words from dataset.\n".format(len(all_words)))
    thumbs_up_mask = np.array(Image.open(os.path.join("data", "img", "thumbs_up_mask.png")))

    cloud_of_words = WordCloud(width=1080,
                                height = 1080,
                                max_font_size = 110,
                                background_color="white",
                                collocations = False,
                                mask = thumbs_up_mask,
                                contour_width = 3,
                                contour_color = (219, 240, 223)).generate(all_words)

    cloud_of_words.recolor(color_func=real_color_func, random_state=3)
    plt.figure(figsize = (12,10))
    plt.imshow(cloud_of_words, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    return cloud_of_words, all_words, len(all_words)

def pareto(dataframe, text_column, max_words):
    all_words = ' '.join([text for text in dataframe[text_column]])

    token_space = tokenize.WhitespaceTokenizer()
    token_phrase = token_space.tokenize(all_words)

    frequency = nltk.FreqDist(token_phrase)

    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
    #print(df_frequency.shape)
    df_frequency_huge = df_frequency.nlargest(columns = "Frequency", n = 200)
    file_name = 'words_frequency_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
    df_frequency_huge.to_csv(file_name)
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = max_words)

    print(df_frequency_huge)
    plt. figure(figsize =(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'gray')
    ax.set(ylabel = "Number of appearances")
    plt.title("Pareto")
    plt.show()
    return len(all_words)

def pareto_tokenized(tokens, max_words):
    # tokens = list()
    # for news_text in dataframe[text_column]:
    #     for word in news_text:
    #         tokens.append(word)

    frequency = nltk.FreqDist(tokens)

    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
    #print(df_frequency.shape)
    df_frequency_huge = df_frequency.nlargest(columns = "Frequency", n = 200)
    file_name = 'words_frequency_' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
    df_frequency_huge.to_csv(file_name)
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = max_words)

    print(df_frequency_huge)
    plt. figure(figsize =(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'gray')
    ax.set(ylabel = "Number of appearances")
    plt.title("Pareto")
    plt.show()
    return len(tokens)

def main():
    # Import CSV dataset file to Pandas DataFrame:
    print("====================================================================")
    print("========================== SCRIPT START ============================\n")
    path = 'data'
    dataset_csv = 'real_and_fake_news_corpus_pt_br.csv'
    print("Importing CSV dataset '{}' from folder '{}':\n".format(dataset_csv, path))
    news = pd.read_csv(os.path.join(path, dataset_csv))
    print("Imported {} instances with {} attributes.".format(news.shape[0], news.shape[1]))

    classification = news["Tag"].replace(["FAKE", "REAL"], [0, 1])
    news["classification"] = classification

    print("The logistic regression resulted in {}% of accuracy.\n".format(100*text_classifier(news, "news_text_normalized", "classification")))
    

    print("====================================================================\n")
    print("Word cloud without any data preprocessing.\n")
    print("Word cloud of all news:\n")
    _, all_words, _ = word_cloud_complete(news, "news_text_normalized", "classification")
    print("\n\n\nWord cloud of real news:\n")
    word_cloud_real(news, "news_text_normalized", "classification")
    print("\n\n\nWord cloud of fake news:\n")
    word_cloud_fake(news, "news_text_normalized", "classification")
    print("\n\n\n====================================================================\n")
    #plt.show(block=False)
    #input('press <ENTER> to continue')
    #plt.savefig('word_cloud.png')

    #Tokenization of dataset
    print("====================================================================")
    print("TOKENIZATION STARTED")
    token_space = tokenize.WhitespaceTokenizer()
    token_phrase = token_space.tokenize(all_words)

    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                              "Frequency": list(frequency.values())})
    print("The 10 most frequent words in this dataset are:\n")
    print(df_frequency.nlargest(columns = "Frequency", n = 10))

    plt. figure(figsize =(12,8))
    ax = sns.barplot(data = df_frequency.nlargest(columns = "Frequency", n = 30), x = "Word", y = "Frequency", color = 'gray')
    ax.set(ylabel = "Number of appearances")
    plt.show()

    pareto(news, "news_text_normalized", 10)

    #Cleaning only stopwords lowercase
    stopwords = nltk.corpus.stopwords.words("portuguese")

    clean_phrase = list()
    for text in news.news_text_normalized:
        new_phrase = list()
        text_words = token_space.tokenize(text)
        for word in text_words:
            if word not in stopwords:
                new_phrase.append(word)
        clean_phrase.append(' '.join(new_phrase))

    news["traitement_1"] = clean_phrase
    
    # Cleaning stopwords lowercase and punctuation
    punctuation_list = list()
    for punct in punctuation:
        punctuation_list.append(punct)
    
    personalized_stopwords = ["“",
                              "”",
                              ",",
                              "”,",
                              '""."',
                              '"),"',
                              '–',
                              'R',
                              '..',
                              '""","',
                              '[...]',
                              ').',
                              '...',
                              '"."""',
                              '),',
                              '".']
    stopwords_with_punctuation = punctuation_list + stopwords + personalized_stopwords
    token_punct = tokenize.WordPunctTokenizer()
    clean_phrase = list()
    for text in news["traitement_1"]:
        new_phrase = list()
        text_words = token_punct.tokenize(text)
        for word in text_words:
            if word not in stopwords_with_punctuation:
                new_phrase.append(word)
        clean_phrase.append(' '.join(new_phrase))

    news["traitement_2"] = clean_phrase

    no_accent = [unidecode.unidecode(text) for text in news["traitement_2"]]
    stopwords_with_punctuation_no_accent = [unidecode.unidecode(text) for text in stopwords_with_punctuation]

    news["traitement_3"] = no_accent

    no_accent_news_text_normalized = [unidecode.unidecode(text) for text in news["news_text_normalized"]]

    news["traitement_4"] = no_accent_news_text_normalized

    processed_phrase = list()

    for text in news["traitement_4"]:
        new_phrase = list()
        text = text.lower()
        text_words = token_punct.tokenize(text)
        for word in text_words:
            if word not in stopwords_with_punctuation_no_accent:
                new_phrase.append(word)
        processed_phrase.append(' '.join(new_phrase))

    news["traitement_4"] = processed_phrase

    # Traitement 5 - Selected stopwords by deseption analisis in portuguese:
    # news["traitement_5"] = no_accent_news_text_normalized
    # selected_stopwords = punctuation_list + stopwords + personalized_stopwords
    #     for text in news["traitement_5"]:
    #     new_phrase = list()
    #     text = text.lower()
    #     text_words = token_punct.tokenize(text)
    #     for word in text_words:
    #         if word not in selected_stopwords:
    #             new_phrase.append(word)
    #     processed_phrase.append(' '.join(new_phrase))

    # news["traitement_5"] = processed_phrase


    pareto(news, "traitement_1", 10)
    print("==========================================================================\n")
    print("Word cloud with preprocessing 1 - simple cleaning of stopwords.\n")
    print("Word cloud of all news:\n")
    _, all_words, _ = word_cloud_complete(news, "traitement_1", "classification")
    print("\n\n\nWord cloud of real news:\n")
    word_cloud_real(news, "traitement_1", "classification")
    print("\n\n\nWord cloud of fake news:\n")
    word_cloud_fake(news, "traitement_1", "classification")
    print("\n\n\n====================================================================\n")
    print("==========================================================================")

    pareto(news, "traitement_2", 10)
    print("==========================================================================\n")
    print("Word cloud with preprocessing 2 - cleaning of stopwords and punctuation.\n")
    print("Word cloud of all news:\n")
    _, all_words, _ = word_cloud_complete(news, "traitement_2", "classification")
    print("\n\n\nWord cloud of real news:\n")
    word_cloud_real(news, "traitement_2", "classification")
    print("\n\n\nWord cloud of fake news:\n")
    word_cloud_fake(news, "traitement_2", "classification")
    print("\n\n\n====================================================================\n")
    print("==========================================================================")

    pareto(news, "traitement_3", 10)
    print("=================================================================================\n")
    print("Word cloud with preprocessing 3 - cleaning of accents, stopwords and punctuation.\n")
    print("Word cloud of all news:\n")
    _, all_words, _ = word_cloud_complete(news, "traitement_3", "classification")
    print("\n\n\nWord cloud of real news:\n")
    word_cloud_real(news, "traitement_3", "classification")
    print("\n\n\nWord cloud of fake news:\n")
    word_cloud_fake(news, "traitement_3", "classification")
    print("\n\n\n==========================================================================\n")
    print("================================================================================")

    pareto(news, "traitement_4", 10)
    print("============================================================================================\n")
    print("Word cloud with preprocessing 4 - cleaning of accents, stopwords, punctuation in lower case.\n")
    print("Word cloud of all news:\n")
    _, all_words, _ = word_cloud_complete(news, "traitement_4", "classification")
    print("\n\n\nWord cloud of real news:\n")
    word_cloud_real(news, "traitement_4", "classification")
    print("\n\n\nWord cloud of fake news:\n")
    word_cloud_fake(news, "traitement_4", "classification")
    print("\n\n\n======================================================================================\n")
    print("============================================================================================")

    print(news.head())

    

if __name__ == '__main__':
    main()
    
# %%
