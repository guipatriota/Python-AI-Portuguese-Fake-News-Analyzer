import pandas as pd
from os import path
import glob

def read_files_to_list(path):
	texts_list = []
	list_of_files = glob.glob(path + '/*.txt')
	for file_name in list_of_files:
		with open(file_name, 'r') as text:
			texts_list.append(text.read())
	return texts_list

true_full_texts_list = read_files_to_list('./data/full_texts/true')
fake_full_texts_list = read_files_to_list('./data/full_texts/fake')
true_normalized_texts_list = read_files_to_list('./data/size_normalized_texts/true')
fake_normalized_texts_list = read_files_to_list('./data/size_normalized_texts/fake')

metadata = ['author',
'news_link',
'category',
'news_date']

print(true_full_texts_list[0],fake_full_texts_list[0])