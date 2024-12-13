import pandas as pd
import csv

df = pd.read_excel('./data/imdb.xlsx')

df[['text', 'class_label']].to_csv('data/imdb.csv',index=False,  quotechar='"', quoting=csv.QUOTE_ALL)