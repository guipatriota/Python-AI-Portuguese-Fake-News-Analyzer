# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import natsort as ns
import numpy as np
import nltk
import os
import fakenewsanalyzerptbr as fna
import string
import unidecode
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nltk.util import ngrams
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt


# %%
path = 'data'
dataset_csv = 'real_and_fake_news_corpus_pt_br.csv'


# %%
news = pd.read_csv(os.path.join(path, dataset_csv))


# %%
news.name = "ALL NEWS"


# %%
news.head(2)


# %%
classification = news["Tag"].replace(["FAKE", "REAL"], [0, 1])
news["classification"] = classification


# %%
_, all_words, len_all_words = fna.word_cloud_complete(news, "news_text_full", "classification")


# %%
_, fake_words, len_fake_words = fna.word_cloud_fake(news, "news_text_full", "classification")


# %%
_, real_words, len_real_words = fna.word_cloud_real(news, "news_text_full", "classification")


# %%
len(real_words)


# %%
len(news.query('classification == 0'))


# %%
news_fake = news.query('classification == 0')
news_real = news.query('classification == 1')
news_fake.head(2)


# %%
news_real.head(2)


# %%
list(news_fake['news_text_full'].head(1))


# %%
list(news_fake['author'].unique())


# %%
news_fake.groupby('author').count()


# %%
news_fake[news_fake.author == 'None'].groupby('author').count()


# %%
news_real[news_real.author == 'None'].groupby('author').count()


# %%
news["author_score"] = 1


# %%
# Criação da coluna "author_score" sendo 1 para autor existente e 0 para notícias sem autor
news["author_score"] = news["author"].replace(["None"], [0])
news["author_score"] = news["author_score"].where(news["author_score"] == 0, 1)


# %%
news.head()


# %%
# Contagem de notícias sem autor
news[news.author_score == 0].groupby('classification').count()


# %%
# Preprocessamento do dataset:


# %%
# Unicodes para retirar:
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
            "\x97": ""
        }
personalized_simbols = ["“",
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
                        '".',
                        'aa']
string.punctuation


# %%
# Conversão da string de pontuações em lista
punctuation_list = list()
for punct in string.punctuation:
    punctuation_list.append(punct)
# Adicionando strings de pontuação que não estão presentes em string.punctuation (“ e ”)
#punctuation = string.punctuation + '“' + '”'
punctuation_simbols_list = punctuation_list + personalized_simbols
punctuation_simbols_list[:10]


# %%
# TESTE - Decodificar texto já sem acentos nem unicode
with open('data/full_texts/fake/2.txt', 'r', encoding='utf8') as text:
    teste1 = unidecode.unidecode(text.read())
    for key in unicodes_to_strip:
        teste1 = teste1.replace(key, unicodes_to_strip[key])
    #teste1 = text.read()
teste1[:150]


# %%
with open(os.path.join('data','stopwords_nltk_ordered.txt'), 'r', encoding='utf8') as text:
    #stop_words_extended = unidecode.unidecode(text.read().splitlines())
    stop_words_extended = text.read().splitlines()
stop_words_extended = set(nltk.tokenize.wordpunct_tokenize(unidecode.unidecode(' '.join(stop_words_extended))))


# %%
clean_phrases = list()
tokens_traitement_1 = list()
#stop_words_nltk = set(nltk.corpus.stopwords.words('portuguese'))
#stop_words_nltk = set(nltk.tokenize.wordpunct_tokenize(unidecode.unidecode(' '.join(nltk.corpus.stopwords.words('portuguese')))))
#stop_words_nltk = set(nltk.tokenize.wordpunct_tokenize(unidecode.unidecode(' '.join(nltk.corpus.stopwords.words('portuguese')))))
stop_words_nltk = stop_words_extended
for text in news.news_text_full:
    # Decodificar texto sem acentos nem unicode
    news_text = unidecode.unidecode(text)
    for key in unicodes_to_strip:
        # Retira \n e \n\n, principalmente, além dos demais unicodes que possam ter sobrado.
        news_text = news_text.replace(key, unicodes_to_strip[key])
    # Retira stopwords:
    filtered_news = [w for w in nltk.tokenize.wordpunct_tokenize(news_text) if not w in stop_words_nltk]
    # Retira pontuação e deixa todas as palavras em minúsculo:
    filtered_news = [word.lower() for word in filtered_news if not word in (punct for punct in punctuation_simbols_list)]
    filtered_news = [w for w in filtered_news if w.isalpha()]
    filtered_news = [w for w in filtered_news if not w in stop_words_nltk]
    tokens_traitement_1.append(len(filtered_news))
    clean_phrases.append(filtered_news)
# Criar lista com frases tokenizadas e tratadas:
#clean_phrases = [s for s in clean_phrases for w in s if not w in stop_words_nltk]
news['traitement_1'] = clean_phrases
# Criar coluna number_of_tokens_traitement_1
news['number_of_tokens_traitement_1'] = tokens_traitement_1


# %%
nltk.tokenize.wordpunct_tokenize("Olá! Eu chamo-me Guilherme.")


# %%
news.traitement_1.head()


# %%
# Reduzir palavras aos seus radicais: STEM
st = nltk.stem.RSLPStemmer()
stem_traitement_1 = list()
for instance in news.traitement_1:
    stem_phrase = list()
    for word in instance:
        stem_phrase.append(st.stem(word))
    stem_traitement_1.append(stem_phrase)
news['traitement_2'] = stem_traitement_1


# %%
news.traitement_2[:2]


# %%
print("Número médio de palavras antes da retirada das stopwords, pontuações e símbolos:\n\n                                                {:.2f} palavras/tokens por notícia\nsendo:\n\n".format(news.number_of_tokens.mean()))
print("Média de palavras das notícias FALSAS:          {:.2f}\n".format(news.groupby('classification').mean()['number_of_tokens'][0]))
print("Média de palavras das notícias VERDADEIRAS:     {:.2f}\n".format(news.groupby('classification').mean()['number_of_tokens'][1]))#print("Sendo {:.2f} palavras em média para notícias falsas e {:.2f} para verdadeiras.\n\n".format(news.groupby('classification').mean()['number_of_tokens'][0], news.groupby('classification').mean()['number_of_tokens'][1]))
print("Já para o primeiro tratamento, restaram:\n                                                {:.2f} palavras em média por notícia, sendo:\n\n".format(news.number_of_tokens_traitement_1.mean()))
print("Média de palavras das notícias FALSAS:          {:.2f}\n".format(news.groupby('classification').mean()['number_of_tokens_traitement_1'][0]))
print("Média de palavras das notícias VERDADEIRAS:     {:.2f}\n".format(news.groupby('classification').mean()['number_of_tokens_traitement_1'][1]))


# %%
news[news.author_score == 0].groupby('classification').count()


# %%
print("NÚMERO DE NOTÍCIAS SEM AUTORIA ASSINADA:\n\n")
print("Falsas:          {}".format(news[news.author_score == 0].groupby('classification').count()['Id'][0]))
print("Verdadeiras:     {}".format(news[news.author_score == 0].groupby('classification').count()['Id'][1]))


# %%
# Retirando pontuação e números. Deixando todas as palavras em minúsculo
# isalpha() retirou palavras com hífem, números e simbolos úteis. Não usar!
# teste2 = [w.lower() for w in filtered_news if w.isalpha()]


# %%
# Criar dataframes FAKE e REAL:
all_news_words = list()
all_real_news_words = list()
all_fake_news_words = list()
news_fake = news.query('classification == 0')
news_fake.name = 'FAKE NEWS'
news_real = news.query('classification == 1')
news_real.name = 'REAL NEWS'


# %%
news_fake['traitement_2'].tail()


# %%
news_fake = news_fake.reset_index()
news_real = news_real.reset_index()


# %%
# TRUNCAR NOTÍCIAS
# Medir tamanhos e podar o maior pelo menor
news_fake['traitement_3'] = ''
news_real['traitement_3'] = ''

for row in range(len(news_fake)):
    if len(news_fake.traitement_2[row]) > len(news_real.traitement_2[row]):
        news_fake.traitement_3[row] = news_fake.traitement_2[row][:len(news_real.traitement_2[row])]
        news_real.traitement_3[row] = news_real.traitement_2[row]
    else:
        news_real.traitement_3[row] = news_real.traitement_2[row][:len(news_fake.traitement_2[row])]
        news_fake.traitement_3[row] = news_fake.traitement_2[row]


# %%
token_count = []
for news_text in news_fake['traitement_3']:
    token_count.append(len(news_text))


# %%
# Unificar dataset:
# token_count
count_fake_real_news = -1
news['traitement_3'] = ''
for row in range(len(news)):
    if row % 2 == 0:
        count_fake_real_news += 1
        news['traitement_3'][row] = news_fake['traitement_3'][count_fake_real_news]
    else:
        news['traitement_3'][row] = news_real['traitement_3'][count_fake_real_news]
    


# %%
news_fake.name = 'FAKE NEWS'
news_real.name = 'REAL NEWS'


# %%
# PARETO
# coluna a ser analisada:
df = news_fake
df_col = 'traitement_3'

fna.pareto_df_tokenized(df, df_col, 10)


# %%

# PARETO
# coluna a ser analisada:
df = news_real
df_col = 'traitement_3'

fna.pareto_df_tokenized(df, df_col, 10)


# %%
# Todas as palavras unificadas
col = 'traitement_3'
# for news_text in news[col]:
#     for word in news_text:
#        all_news_words.append(word)
all_fake_news_words = []
all_real_news_words = []
for news_text in news_fake[col]:
    for word in news_text:
       all_fake_news_words.append(word)

for news_text in news_real[col]:
    for word in news_text:       
       all_real_news_words.append(word)


# %%
len(all_fake_news_words)


# %%
len(all_real_news_words)


# %%
bigrams_to_suppress = [
    ('ex', 'presid'),
    ('lav','jat'),
    ('segund','feir'),
    ('terc','feir'),
    ('quart','feir'),
    ('quint','feir'),
    ('sext','feir'),
    ('michel','tem'),
    ('sergi', 'mor'),
    ('dilm', 'rousseff'),
    ('eduard', 'cunh')]


# %%
bigrams_fake = list(ngrams(all_fake_news_words, 2))
bigrams2 = []

for bigram in bigrams_fake:
    bigram2 = []
    if not bigram in (bigram for bigram in bigrams_to_suppress):
        for word in bigram:
            bigram2.append(word)
        bigrams2.append(' '.join(bigram2))

bigrams_fake = bigrams2
bigrams_fake[:3]


# %%
len(bigrams_fake)


# %%
def word_cloud_bigram(bigrams, mask):
    len_bigrams = len(bigrams)
    print("   * Were computed a total of {} bigrams from dataset.\n".format(len_bigrams))
    
    mask_default = "cloud_mask.png"
    #if os.path.join("data", "img", mask)
        
    def color_function(mask):
        if mask == "mapa_brasil_mask.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(190, 40%%, %d%%)" % random.randint(30, 60) #sky
            color_cont = (219, 236, 240)
        elif mask == "thumbs_down_mask_3.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(0, 80%%, %d%%)" % random.randint(30, 60) #fake
            color_cont = (250, 209, 209)
        elif mask == "thumbs_up_mask.png":
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(130, 40%%, %d%%)" % random.randint(30, 60) #real
            color_cont = (219, 240, 223)
        else:
            def color_func(word, font_size, position, orientation, random_state=None,**kwargs):
                return "hsl(0, 0%%, %d%%)" % random.randint(60, 100) #grey
            color_cont = (219, 236, 240)
        return color_func, color_cont

    color_function, color_cont = color_function(mask)

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(bigrams)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)

    mask = np.array(Image.open(os.path.join("data", "img", mask)))

    # cloud_of_words = WordCloud(width = 1080,
    #                            height = 1080,
    #                            max_font_size = 110,
    #                            collocations = False,
    #                            mask = mask,
    #                            background_color = "white",
    #                            contour_width = 3,
    #                            contour_color = (219, 236, 240)).generate_from_frequencies(words_dict)
    # cloud_of_words.recolor(color_func=color_func, random_state=3)

    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    cloud_of_words = WordCloud(height=1080,
                               width=1080,
                               max_font_size = 110,
                               collocations = False,
                               background_color = "white",
                               mask = mask,
                               contour_width = 3,
                               contour_color = color_cont)
    cloud_of_words.generate_from_frequencies(words_dict)
    cloud_of_words.recolor(color_func=color_function, random_state=3)
    #plt.title('Most frequently occurring bigrams connected by same colour and font size')
    plt.figure(figsize = (12, 10))
    plt.imshow(cloud_of_words, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()
    return cloud_of_words, len_bigrams


# %%
bigrams_real = list(ngrams(all_real_news_words, 2))
bigrams2 = []

for bigram in bigrams_real:
    bigram2 = []
    if not bigram in (bigram for bigram in bigrams_to_suppress):
        for word in bigram:
            bigram2.append(word)
        bigrams2.append(' '.join(bigram2))

bigrams_real = bigrams2
bigrams_real[:3]


# %%
word_cloud_bigram(bigrams_fake, "thumbs_down_mask_3.png")


# %%
word_cloud_bigram(bigrams_real, "thumbs_up_mask.png")


# %%
fna.pareto_all_tokenized(bigrams_fake, 'FAKE', 10)


# %%
fna.pareto_all_tokenized(bigrams_real, 'REAL', 10)


# %%
news.head()


# %%
processed_phrase = list()
for phrase in news['traitement_3']:
    processed_phrase.append(' '.join(phrase))

news['traitement_4'] = processed_phrase


# %%
accuracy_bag_traitement_4 = fna.text_classifier(news, 'traitement_4', 'classification')


# %%
accuracy_bag_news_text_full = fna.text_classifier(news, 'news_text_full', 'classification')


# %%
accuracy_bag_news_text_normalized = fna.text_classifier(news, 'news_text_normalized', 'classification')


# %%
# Criação do modelo de regressão logística e de TF-IDF:
from sklearn.feature_extraction.text import TfidfVectorizer
regressao_logistica = LogisticRegression(solver = 'lbfgs')
tfidf = TfidfVectorizer(lowercase = False, max_features = 500)


# %%
# Acurácia do TF-IDF para os textos completos sem tratamento:
tfidf_text_full = tfidf.fit_transform(news["news_text_full"])

treino, teste, classe_treino, classe_teste = train_test_split(tfidf_text_full,
                                                              news["classification"],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia__tfidf_text_full = regressao_logistica.score(teste, classe_teste)
print(acuracia__tfidf_text_full)

#Para visualizar a matriz de frequências:
caracteristicas = tfidf.fit_transform(news["news_text_full"])
pd.DataFrame(
     caracteristicas.todense(),
     columns = tfidf.get_feature_names())


# %%
tfidf_text_normalized = tfidf.fit_transform(news["news_text_normalized"])

treino, teste, classe_treino, classe_teste = train_test_split(tfidf_text_normalized,
                                                              news["classification"],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_text_normalized = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_text_normalized)


# %%
tfidf_traitement_4 = tfidf.fit_transform(news["traitement_4"])

treino, teste, classe_treino, classe_teste = train_test_split(tfidf_traitement_4,
                                                              news["classification"],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_traitement_4 = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_traitement_4)


# %%
# ATENÇÃO!!! Este processo pode causar estouro de memória quando fizermos a
# criação do pandas DataFrame (faremos: vetor_tfidf.todense()).
# Isto ocorre pois o vetor possui 7200 linhas (quantidades de texto) e até
# 951841 colunas (os ngramas possíveis de 1 a 3). Como dado está no formato
# float64, que possui 64 bits (8 Bytes), o cálculo de memória necessário é:
# [7200 * 8/(1024^3)] * 951841 = 51,06 GB de memória necessários.
# Para reduzir esta quantidade necessária, vamos utilizar o parâmetro
# max_features da função TfidfVectorizer() com um valor que resulte em menos
# da metade da memória disponível no computador, para não termos problemas.
# Novo cálculo:
# max_features = 8 GB / [7200 * 8/(1024^3)] = 149130
#
# Outra solução é não usar o max_features e ao criar o DataFrame Pandas, usar
# pd.DataFrame.sparse.from_spmatrix(), para criar um dataframe de matriz esparsa

#tfidf = TfidfVectorizer(lowercase = False, ngram_range = (1,3), max_features=149130)

tfidf = TfidfVectorizer(lowercase = False, ngram_range = (1,3))

vetor_tfidf = tfidf.fit_transform(news["traitement_4"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              news["classification"],
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams_traitement_4 = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams_traitement_4)


# %%
classe_teste


# %%
#Para visualizar a matriz de frequências:
#pd.DataFrame(
#     vetor_tfidf.todense(),
#     columns = tfidf.get_feature_names())

# %%
#Para visualizar a matriz de frequências com matriz esparsa:
# Economia de memória!

pd.DataFrame.sparse.from_spmatrix(
     vetor_tfidf,
     columns = tfidf.get_feature_names_out())

# %%
pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T, #coef_[0] é o peso de cada item e .T realiza a transposição da matriz
    index = tfidf.get_feature_names_out()
)

pesos.nlargest(150,0) #mostra os 10 maiores pesos (sentimentos positivos)


# %%
pesos.nsmallest(150,0) #mostra os 10 menores pesos (sentimentos negativos)


# %%



# %%
# predicts1 = pd.DataFrame(regressao_logistica.predict_proba(tfidf.fit_transform(news["traitement_4"])), columns = ['FAKE', 'REAL'])
predicts1 = pd.DataFrame(regressao_logistica.predict_proba(teste), columns = ['FAKE', 'REAL'])


# %%
classe_teste


# %%
predicts1


# %%
regressao_logistica.predict_proba(teste)




# %%
index_teste = classe_teste.index

# %%
index_teste[1]

# %%
predicts2_list = list()
for row in range(len(predicts1)):
    if news.author_score[index_teste[row]] == 0:
        predicts2_list.append((predicts1['FAKE'][row]*0.8 + 0.2, 1 - (predicts1['FAKE'][row]*0.8 + 0.2)))
    else:
        predicts2_list.append((1 - (predicts1['REAL'][row]*0.8 + 0.2), predicts1['REAL'][row]*0.8 + 0.2))
predicts2 = pd.DataFrame(predicts2_list, columns=['FAKE', 'REAL'])
predicts2.head()


# %%
predicts2['predict'] = 0
for row in range(len(predicts2)):
    if predicts2['FAKE'][row] >= predicts2['REAL'][row]:
        predicts2['predict'][row] = 0
    else:
        predicts2['predict'][row] = 1


# %%
predicts2.predict.to_numpy()


# %%
classe_teste


# %%
from sklearn.metrics import accuracy_score
acuracia_tfidf_ngrams_mais_registro_autor = accuracy_score(classe_teste, predicts2.predict.to_numpy(), sample_weight=None)
acuracia_tfidf_ngrams_mais_registro_autor


# %%
print('Acurácia para Bag of Words em news_text_full:             {:.2f}%'.format(100*accuracy_bag_news_text_full))
print('Acurácia para Bag of Words emn ews_text_normalized:       {:.2f}%'.format(100*accuracy_bag_news_text_normalized))
print('Acurácia para Bag of Words em traitement_4:               {:.2f}%'.format(100*accuracy_bag_traitement_4))
print("Acurácia para TF-IDF em news_text_full:                   {:.2f}%".format(100*acuracia__tfidf_text_full))
print("Acurácia para TF-IDF em news_text_normalized:             {:.2f}%".format(100*acuracia_tfidf_text_normalized))
print("Acurácia para TF-IDF em traitement_4:                     {:.2f}%".format(100*acuracia_tfidf_traitement_4))
print("Acurácia para TF-IDF com 1, 2 e 3 ngrams em traitement_4: {:.2f}%".format(100*acuracia_tfidf_ngrams_traitement_4))
print("\nAcurácia para TF-IDF com 1, 2 e 3 ngrams em \ntraitement_4 mais avaliação de autor existente ou não:    {:.2f}%".format(100*acuracia_tfidf_ngrams_mais_registro_autor))
print("\n\nProcedimento: com os textos completos, tokenizer, retirar os \
    acentos e números, deixar tudo em minúsculas, retirar stopwords e \
    pontuações, deixar palavras apenas com radical, realizar truncamento dos \
    pares de notícias verdadeiras com falsas para normalizar quantidade de \
    palavras, remontar as notícias em string e criar coluna no dataframe para \
    o resultado deste pré-processamento. Criar coluna do DF com informação da \
    existência ou não de autoria da notícia (0 não e 1 sim). Criar matriz de \
    frequências TF-IDF com ngramas de 1 a 3 palavras. Usar a função \
    train_test_split do Scikit Learn para dividir o corpus pré-tratado em 75% \
    para treinamento e 25% para testes de precisão (usado random_state = 42). \
    Fazer regressão logística com solver = 'lbfgs'. Realizar predição dos \
    textos de teste com o método predict_proba, que retornará a porcentagem \
    predita para fake e para real em um array. Com a informação da existência \
    do autor ou não, recalcular a porcentagem com peso de 20% para a existência \
    do autor em favor da notícia ser verdadeira e 20% menos em caso de ausência \
    de autor. Verificar qual porcentagem foi maior que 50% para sinalizar 0 à \
    predição FAKE e 1 REAL. Importante observar que a fórmula usada para \
    realizar a correção da porcentagem de predição é dada por peso, sendo 80% \
    de peso para o valor predito pela regressão logística e 20% para o autor \
    existente ou menos 20% para o não existente, premiando a existência do \
    autor e punindo quando não possui, aumentando a distância das estimativas. \
    Por fim, com as porcentagens recalculadas para cada texto de teste, foi \
    usada a função accuracy_score da biblioteca Scikit Learn para calcular a \
    nova acurácia geral do algoritmo.\n \
    É importante observar que este algoritmo dá um passo a mais em direção às \
    técnicas de identificação de notícias falsas criadas pelo professor Gabriel, \
    levando em conta dois de dez passos, os da verificação da fonte. Por este \
    motivo, a proporção de 20%/80% foi escolhido para os cálculos da média \
    ponderada. Ainda, salientamos que nenhuma avaliação da plausabilidade ou \
    veracidade de fatos específicos foi adicionada, o que certamente poderá \
    causar uma performance inferior ao calculado em um caso de uso real do \
    mesmo. A acurácia de 97,83% é realmente um valor muito superior aos \
    anteriormente obtidos e acreditamos que isto possa ser reflexo da grande \
    quantidade de erros ortográficos nas notícias falsas presentes neste \
    dataset, o que não expõe uma falha do mesmo, mas sim um padrão das fontes \
    de criação de notícias falsas.")


# %%
regressao_logistica.predict(teste)


# %%
_, all_words, len_all_words = fna.word_cloud_complete(news, "traitement_4", "classification")


# %%
_, fake_words, len_fake_words = fna.word_cloud_fake(news, "traitement_4", "classification")


# %%
_, real_words, len_real_words = fna.word_cloud_real(news, "traitement_4", "classification")

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################






# %% [markdown]
# Testes de context free grammar (CFG):

# %%
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
from nltk.corpus import floresta

import nltk
# %%
grammar = CFG.fromstring(demo_grammar)
print(grammar)

# %%
demo_grammar

# %%
floresta.tagged_words()

# %%
floresta.words()

# %%
def simplifica_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t
    
# %%
twords = floresta.tagged_words()
twords = [(w.lower(), simplifica_tag(t)) for (w,t) in twords]
twords[:10]

# %%
words = floresta.words()
len(words)

# %%
fd = nltk.FreqDist(words)
len(fd)

# %%
fd.max()

# %%
tags = [simplifica_tag(tag) for (word,tag) in floresta.tagged_words()]
tags

# %%
fd = nltk.FreqDist(tags)
tags

# %%
list(fd.keys())[:20]
# %%
floresta.sents()

# %%
floresta.tagged_sents()

# %%
floresta.parsed_sents()

# %%
psents = floresta.parsed_sents()
psents[142].draw()
# %%
floresta.parsed_sents()[538].draw()
# %%
grammar = """"""""

ruleset = set(rule for tree in psents[:10] 
           for rule in tree.productions())

for rule in ruleset:
        grammar += str(rule).format

# %%
production = []
for tree in psents[:10]:
    production += tree.productions()
S = nltk.Nonterminal("S")
grammar = nltk.induce_pcfg(S, production)

# %%
grammar1 = nltk.CFG.fromstring(grammar)
# %%
texto = 'Teste de sentença completa.'
sentencas = nltk.sent_tokenize(texto)
print(sentencas)
palavras = nltk.word_tokenize(texto)
print(palavras)

# %%
nltk.pos_tag(texto)

# %%
nltk.pos_tag(sentencas)

# %%
nltk.pos_tag(palavras)
# %%
nltk.pos_tag(palavras, tagset='universal')

# %%
nltk.pos_tag(palavras, tagset='')

# %%
nltk.help.floresta_tagset()
# %%
nltk.corpus.mac_morpho.tagged_words()

# %%

# OBSERVAÇÃO: exemplo de desempacotamento das tuplas de palavra e tag:
[[word for word in sents] for sents in floresta.tagged_sents()[1]]

# %%
# Criação de tagged sentences limpas a partir das tagged_sents sujas de floresta:

floresta_tagged_sents_original = floresta.tagged_sents()

floresta_tagged_sents_limpa = [[(w.lower(), simplifica_tag(t)) for (w,t) in [word for word in sents]] for sents in floresta_tagged_sents_original]

floresta_tagged_sents_limpa[:10]

# %%
# Comparação dos resultados:
print('Tagged Sentences original:\n',floresta_tagged_sents_original[0],'\n\n')
print('Tagged Sentences limpa:\n', floresta_tagged_sents_limpa[0])

# %%
floresta.parsed_sents()[5].draw()
# %%

