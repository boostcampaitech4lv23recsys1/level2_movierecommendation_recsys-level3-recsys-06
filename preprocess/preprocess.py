import pandas as pd

path = "../../data/train/"
director = pd.read_csv(path+"directors.tsv",sep='\t')
genre = pd.read_csv(path+"genres.tsv",sep='\t')
title = pd.read_csv(path+"titles.tsv",sep='\t')
train_rating = pd.read_csv(path+"train_ratings.csv",sep=',')
writer=pd.read_csv(path+"writers.tsv",sep='\t')
year=pd.read_csv(path+"years.tsv",sep='\t')
