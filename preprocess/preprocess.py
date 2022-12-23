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
    # train_ratings.csv 파일 처리
    train_rating['date'] = pd.to_datetime(train_rating['time'], unit='s')
    train_rating = train_rating.sort_values(by = ['user','time'], axis = 0)
    # watch_year, watch_month, watch_hour, watch_day
    train_rating['watch_year'] = train_rating['date'].dt.strftime('%Y')
    train_rating['watch_month'] = train_rating['date'].dt.strftime('%m')
    train_rating['watch_day'] = train_rating['date'].dt.strftime('%d')
    train_rating['watch_hour'] = train_rating['date'].dt.strftime('%H')
    train_rating['watch_gap'] = train_rating.date.diff().dt.days

    first_idx = list(train_rating.groupby('user').apply(lambda x: x.first_valid_index()))
    train_rating.iloc[first_idx , -1] = 0
    
    # year.tsv 처리
    year.rename(columns = {'year':'release_year'},inplace=True)
    trainyear_df = pd.merge(train_rating, year, on=['item'], how='left')

    # year 결측치 처리 
    no_year_item = trainyear_df[trainyear_df.release_year.isnull()].item.unique()
    no_year_df = title[title['item'].isin(no_year_item)]
    year_na_dict = dict(zip(list(no_year_df.item), list(no_year_df.title.str[-5:-1].astype(np.float64))))
    no_year_fill = trainyear_df[trainyear_df['item'].isin(no_year_item)].index
    for index in no_year_fill:
        trainyear_df.at[index,'release_year'] = year_na_dict[trainyear_df.iloc[index]['item']]
    
    # 피처 생성
    trainyear_df['watch_year_int'] = trainyear_df['watch_year'].astype(int)
    trainyear_df['since_release'] = trainyear_df['watch_year_int'] - trainyear_df['release_year']
    trainyear_df.drop(columns='watch_year_int',inplace=True)

    def release_year_mapping(year, gap) :
        if year <= 1930 : return 0
        else : return (year - 1930) // gap + 1

    trainyear_df['categorized_year_gap5'] = trainyear_df['release_year'].apply(release_year_mapping, args=(5,))
    trainyear_df['categorized_year_gap10'] = trainyear_df['release_year'].apply(release_year_mapping, args=(10,))

    trainyear_df['release_year'] = trainyear_df['release_year'].apply(int).apply(str)

    return trainyear_df


def make_maniatic_feature() :

    # 유저별로 favorite_genre','maniatic' 피처 생성
    # genre.csv를 불러와서 멀티핫인코딩
    temp = pd.get_dummies(genre) 
    genre_multihot = temp.groupby('item').sum().reset_index()

    for col in genre_multihot.columns[1:] :
        genre_multihot.rename(columns = {col : col[6:]}, inplace = True)

    useritem_genre = pd.merge(train_rating.iloc[:,:2], genre_multihot, on=['item'], how='left')
    useritem_count = useritem_genre.groupby('user')['item'].count().reset_index()
    user_genre = useritem_genre.groupby('user').sum().reset_index()
    
    user_genre['favorite_genre'] = user_genre.iloc[:, 2:].idxmax(axis=1)
    user_genre['maniatic'] = user_genre.iloc[:, 2:].max(axis=1) / useritem_count['item']
    user_genre = user_genre[['user','favorite_genre','maniatic']]

    return user_genre