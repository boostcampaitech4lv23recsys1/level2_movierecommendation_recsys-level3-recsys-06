import pandas as pd
import numpy as np
import os

path = "../../data/train/"
director = pd.read_csv(path+"directors.tsv",sep='\t')
genre = pd.read_csv(path+"genres.tsv",sep='\t')
title = pd.read_csv(path+"titles.tsv",sep='\t')
train_rating = pd.read_csv(path+"train_ratings.csv",sep=',')
writer = pd.read_csv(path+"writers.tsv",sep='\t')
year = pd.read_csv(path+"years.tsv",sep='\t')

def make_train_ratings(): 
    train_df['date'] = pd.to_datetime(train_df['time'], unit='s')
    train_df = train_df.sort_values(by = ['user','time'], axis = 0)
    # watch_year, watch_month, watch_hour, watch_day
    train_df['watch_year'] = train_df['date'].dt.strftime('%Y')
    train_df['watch_month'] = train_df['date'].dt.strftime('%m')
    train_df['watch_day'] = train_df['date'].dt.strftime('%d')
    train_df['watch_hour'] = train_df['date'].dt.strftime('%H')
    train_df['watch_gap'] = train_df.date.diff().dt.days

    first_idx = list(train_df.groupby('user').apply(lambda x: x.first_valid_index()))
    train_df.iloc[first_idx , -1] = 0


