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

test = pd.DataFrame([1,2,3])

# %%
