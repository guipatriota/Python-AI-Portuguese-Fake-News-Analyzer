#%%
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk import tokenize

from string import punctuation

from wordcloud import WordCloud
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

def word_cloud_complete(dataframe, text_column, classification_column):
    #%matplotlib inline
    all_words = ' '.join([text for text in dataframe[text_column]])
    #print(all_words[:3])
    print("   * Were computed a total of {} words from dataset.\n".format(len(all_words)))
    cloud_of_words = WordCloud(width = 800,
                              height = 500,
                              max_font_size = 110,
                              collocations = False).generate(all_words)
    plt.figure(figsize = (12, 10))
    plt.imshow(cloud_of_words, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    return cloud_of_words, all_words

def word_cloud_fake(dataframe, text_column, classification_column):
  news = dataframe.query(classification_column + " == 0")
  all_words = ' '.join([text for text in news[text_column]])

  cloud_of_words = WordCloud(width=800,
                                height = 500,
                                max_font_size = 110,
                                background_color="darkred",
                                collocations = False).generate(all_words)

  plt.figure(figsize = (12,10))
  plt.imshow(cloud_of_words, interpolation = 'bilinear')
  plt.axis('off')
  plt.show()
  return cloud_of_words, all_words

def word_cloud_real(dataframe, text_column, classification_column):
  news = dataframe.query(classification_column + " == 1")
  all_words = ' '.join([text for text in news[text_column]])

  cloud_of_words = WordCloud(width=800,
                                height = 500,
                                max_font_size = 110,
                                background_color="white",
                                collocations = False).generate(all_words)

  plt.figure(figsize = (12,10))
  plt.imshow(cloud_of_words, interpolation = 'bilinear')
  plt.axis('off')
  plt.show()
  return cloud_of_words, all_words

def pareto(dataframe, text_column, max_words):
  all_words = ' '.join([text for text in dataframe[text_column]])

  token_space = tokenize.WhitespaceTokenizer()
  token_phrase = token_space.tokenize(all_words)

  frequency = nltk.FreqDist(token_phrase)

  df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
  df_frequency = df_frequency.nlargest(columns = "Frequency", n = max_words)
  plt. figure(figsize =(12,8))
  ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'gray')
  ax.set(ylabel = "Number of appearances")
  plt.title("Pareto")
  plt.show()

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
    _, all_words = word_cloud_complete(news, "news_text_normalized", "classification")
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
    
    stopwords_with_punctuation = punctuation_list + stopwords
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

    pareto(news, "traitement_1", 10)
    pareto(news, "traitement_2", 10)

    print(news.head())

if __name__ == '__main__':
    main()
    
# %%
