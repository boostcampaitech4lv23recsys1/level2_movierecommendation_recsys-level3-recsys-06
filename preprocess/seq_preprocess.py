import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
import time


path = "../data/train/"
director = pd.read_csv(path+"directors.tsv",sep='\t')
genre = pd.read_csv(path+"genres.tsv",sep='\t')
title = pd.read_csv(path+"titles.tsv",sep='\t')
train_rating = pd.read_csv(path+"train_ratings.csv",sep=',')
writer = pd.read_csv(path+"writers.tsv",sep='\t')
year = pd.read_csv(path+"years.tsv",sep='\t')

item_label = LabelEncoder()
item_label.fit(train_rating['item'])

def make_train_ratings(train_rating):
    t_s = time.time()
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
    print("complete train_ratings preprocessing, execution time {:.2f} s".format(time.time() - t_s))
    return trainyear_df


def make_maniatic_feature() :
    t_s = time.time()
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
    print("complete user_genre preprocessing, execution time {:.2f} s".format(time.time() - t_s))
    return user_genre

def user_genre_and_train_rating_preprocess(): 

    user_genre = make_maniatic_feature() 
    trainyear_df = make_train_ratings(train_rating)

    # label Encoding
    le_genre = LabelEncoder()
    le_genre.fit(user_genre['favorite_genre'])
    user_genre['favorite_genre_label'] = le_genre.fit_transform(user_genre['favorite_genre'])

    le_user = LabelEncoder()
    le_user.fit(user_genre['user'])
    user_genre['user_label'] = le_user.transform(user_genre['user'])
    trainyear_df['user_label'] = le_user.transform(trainyear_df['user'])

    year_concat = pd.concat([trainyear_df['watch_year'], trainyear_df['release_year']],axis=0)
    le_year = LabelEncoder()
    le_year.fit(year_concat)
    trainyear_df['watch_year_label'] = le_year.transform(trainyear_df['watch_year'])

    le_watch_month = LabelEncoder()
    trainyear_df['watch_month_label'] = le_watch_month.fit_transform(trainyear_df['watch_month'])
    le_watch_day = LabelEncoder()
    trainyear_df['watch_day_label'] = le_watch_day.fit_transform(trainyear_df['watch_day'])
    le_watch_hour = LabelEncoder()
    trainyear_df['watch_hour_label'] = le_watch_hour.fit_transform(trainyear_df['watch_hour'])

    trainyear_df['release_year_label'] = le_year.transform(trainyear_df['release_year'])
    le_c_year_gap5 = LabelEncoder()
    trainyear_df['categorized_year_gap5_label'] = le_c_year_gap5.fit_transform(trainyear_df['categorized_year_gap5'])
    le_c_year_gap10 = LabelEncoder()
    trainyear_df['categorized_year_gap10_label'] = le_c_year_gap10.fit_transform(trainyear_df['categorized_year_gap10'])
    return user_genre,trainyear_df, le_user

def make_popular_director():
    director_concat_data = train_rating.merge(director, how='left',on=['item'])
    popular_director=director_concat_data['director'].value_counts()
    return popular_director
    
def make_popular_writer():
    writer_concat_data = train_rating.merge(writer,how='left',on=['item'])
    popular_writer=writer_concat_data['writer'].value_counts()
    return popular_writer

def make_main_director():
    director_concat_data = train_rating.merge(director, how='left',on=['item'])
    popular_director=director_concat_data['director'].value_counts()
    director_dict=dict(director.groupby('item')['director'].value_counts())
    item_by_dir={}
    main_dir_dict = {}

    for item,directors in director_dict:
        item_by_dir[item] = item_by_dir.get(item,[])+[directors]
    for movie,directors in item_by_dir.items():
        count=0
        main_director=""
        for content in directors:
            if count < popular_director[content]:
                main_director=content
        main_dir_dict[movie] = main_director
    main_dir_frame=pd.DataFrame(main_dir_dict.items(),columns=['item','main_director'])
    return main_dir_frame

def make_main_writer():
    popular_writer = make_popular_writer()
    writer_dict=dict(writer.groupby('item')['writer'].value_counts())
    item_by_writer = {}
    main_writer_dict={}

    for item,writers in writer_dict:
        item_by_writer[item] = item_by_writer.get(item,[])+[writers]
    for movie,writers in item_by_writer.items():
        count=0
        main_writer=""
        for content in writers:
            if count < popular_writer[content]:
                main_writer=content
        main_writer_dict[movie] = main_writer
    main_writer_frame=pd.DataFrame(main_writer_dict.items(),columns=['item','main_writer'])
    return main_writer_frame

def make_other_director():
    # train_rating에 있지만 director에는 없는 영화의 감독은 결측치이기 때문에 그것을 채워줌
    other_movie=list(set(train_rating['item'].unique())-set(director['item'].unique()))
    other_movie_dict={}
    for i in other_movie:
        other_movie_dict[i]='other_d'
    other_director_frame = pd.DataFrame(other_movie_dict.items(),columns=['item','director'])
    director_total=pd.concat([director,other_director_frame],axis=0)
    director_total=director_total.sort_values(by='item')
    # director_total.to_csv("director_total.tsv",sep='\t',index=False)
    return director_total

def make_other_writer():
    # train_rating에 있지만 writer에는 없는 영화의 작가은 결측치이기 때문에 그것을 채워줌
    other_writer_movie=list(set(train_rating['item'].unique())-set(writer['item'].unique()))

    other_writer_dict={}
    for i in other_writer_movie:
        other_writer_dict[i]='other_w'
    other_writer_frame = pd.DataFrame(other_writer_dict.items(),columns=['item','writer'])
    writer_total=pd.concat([writer,other_writer_frame],axis=0)
    writer_total=writer_total.sort_values(by='item')
    # writer_total.to_csv("writer_total.tsv",sep='\t',index=False)
    return writer_total

def director_preprocess():
    director_total = make_other_director()
    main_director = make_main_director()
    director_main=director_total.merge(main_director, how='left',on=['item'])
    director_main['main_director']=director_main['main_director'].fillna('other_d')
    return director_main

def writer_preprocess():
    writer_total = make_other_writer()
    main_writer = make_main_writer()
    writer_main=writer_total.merge(main_writer, how='left',on=['item'])
    writer_main['main_writer']=writer_main['main_writer'].fillna('other_w')
    return writer_main

def writer_fillna():
    director_main = director_preprocess()
    writer_main = writer_preprocess()
    main_directors = {}
    main_director_dict = dict(director_main.groupby('item')['main_director'].value_counts())
    for key,value in main_director_dict:
        main_directors[key] = main_directors.get(key,"")+value
    main_writers = {}
    main_writer_dict = dict(writer_main.groupby('item')['main_writer'].value_counts())
    for key,value in main_writer_dict:
        main_writers[key] = main_writers.get(key,"")+value
    for key in main_directors:
        if main_writers[key]=='other_w':
            if main_directors[key] !='other_d':
                main_writers[key] = main_directors[key]
    writer_frame = pd.DataFrame(main_writers.items(),columns=['item','main_writer'])
    writer_main=writer_main[['item','writer']].merge(writer_frame, how='left',on=['item'])
    return writer_main


def director_and_writer_labeling():
    director_main = director_preprocess()
    writer_main = writer_preprocess()
    writer_main = writer_fillna()
    le_director_and_writer=LabelEncoder()
    total_label = le_director_and_writer.fit(pd.concat([director_main['director'],writer_main['writer']],axis=0))
    director_main['item_label'] = item_label.transform(director_main['item']) 
    director_main['director_label'] = total_label.transform(director_main['director'])
    director_main['main_director_label'] = total_label.transform(director_main['main_director'])
    writer_main['item_label'] = item_label.transform(writer_main['item']) 
    writer_main['writer_label'] = total_label.transform(writer_main['writer'])
    writer_main['main_writer_label'] = total_label.transform(writer_main['main_writer'])


    return director_main,writer_main

def director_and_writer_preprocess():
    t_s = time.time()
    director_main,writer_main=director_and_writer_labeling()
    director_list = {}
    main_director_list = {}
    for i in range(len(director_main)):
        tmp = director_main.iloc[i]
        director_list[tmp['item_label']]=director_list.get(tmp['item_label'],[])+[tmp['director_label']]
        main_director_list[tmp['item_label']]=[tmp['main_director_label']]
    writer_list = {}
    main_writer_list = {}
    for i in range(len(writer_main)):
        tmp = writer_main.iloc[i]
        writer_list[tmp['item_label']]=writer_list.get(tmp['item_label'],[])+[tmp['writer_label']]
        main_writer_list[tmp['item_label']]=[tmp['main_writer_label']]
    print("complete director and writer preprocessing, execution time {:.2f} s".format(time.time() - t_s))
    return director_list,writer_list,main_director_list,main_writer_list

def title_preprocess():
    t_s = time.time()
    # title[title['title']=='War of the Worlds (2005)']
    # director[director['item']==34048]  # 스필버그 영화 => 더 유명한 우주전쟁
    # item_id가 64997인 녀석은 안 유명한 우주전쟁
    # title[title['item']==64997].title = 'War of the Worlds_B (2005)'
    title.at[1926, 'title'] = 'War of the Worlds_B (2005)'
    title['item_label'] = item_label.transform(title['item'])
    title_le = LabelEncoder()
    title_label=title_le.fit(title['title'])
    title['title_label'] = title_le.transform(title['title'])
    title_list={}
    for i in range(len(title)):
        tmp = title.iloc[i]
        title_list[tmp['item_label']]=[tmp['title_label']]
    print("complete title preprocessing, execution time {:.2f} s".format(time.time() - t_s))
    return title_list

def genre_preprocess():
    t_s = time.time()
    genre_le=LabelEncoder()
    genre_label=genre_le.fit(genre['genre'])
    genre['genre_label'] = genre_le.transform(genre['genre'])
    genre['item_label'] = item_label.transform(genre['item'])
    genre_list={}
    for i in range(len(genre)):
        tmp = genre.iloc[i]
        genre_list[tmp['item_label']]=genre_list.get(tmp['item_label'],[])+[tmp['genre_label']]
    print("complete genre preprocessing, execution time {:.2f} s".format(time.time() - t_s))
    return genre_list

def total_preprocess(data_path):
    data_path = os.path.join(data_path, 'seq_data')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(data_path+'/seq_preprocessed_data.p'):
        print('preprocessed_data already exists')
    
    else: 
        t_s = time.time()
        total_dict = {}
        user_genre,train_edit, user_label = user_genre_and_train_rating_preprocess()
        director_list,writer_list,main_director_list,main_writer_list=director_and_writer_preprocess()
        
        train_edit['item_label'] = item_label.transform(train_edit['item'])
        title_list =title_preprocess()
        genre_list=genre_preprocess()
        for i in tqdm(range(len(train_edit))):
            user_dict={}
            user_tmp = train_edit.iloc[i]
            user=user_tmp['user_label']
            movie=user_tmp['item_label']
            user_dict['item'] = [movie]
            user_dict['watch_year_label'] = [user_tmp['watch_year_label']]
            user_dict['watch_month_label'] = [user_tmp['watch_month_label']]
            user_dict['watch_hour_label'] = [user_tmp['watch_hour_label']]
            user_dict['watch_day_label'] = [user_tmp['watch_day_label']]
            user_dict['watch_gap'] = [user_tmp['watch_gap']]
            user_dict['favorite_genre_label'] = [user_genre[user_genre['user_label']==user]['favorite_genre_label'].values[0]]
            user_dict['maniatic'] = [user_genre[user_genre['user_label']==user]['maniatic'].values[0]]
            user_dict['release_year_label'] = [user_tmp['release_year_label']]
            user_dict['since_release'] = [user_tmp['since_release']]
            user_dict['categorized_year_gap5_label'] = [user_tmp['categorized_year_gap5_label']]
            user_dict['categorized_year_gap10_label'] = [user_tmp['categorized_year_gap10_label']]
            user_dict['title_label']=title_list[movie]
            user_dict['director_label'] = director_list[movie]
            user_dict['writer_label']= writer_list[movie]
            user_dict['main_director_label']=main_director_list[movie]
            user_dict['main_writer_label']=main_writer_list[movie]
            user_dict['genre_label']=genre_list[movie]
            total_dict[user] = total_dict.get(user,[])+[user_dict]
        print("complete preprocessing, execution time {:.2f} s".format(time.time() - t_s))

        with open(data_path+"preprocessed_data.p","wb") as file:
            pickle.dump(total_dict,file)
        
        labels = [user_label, item_label]


        if os.path.exists(data_path+"/seq_labels.p"):
            os.remove(data_path+"/seq_labels.p")
        
        
        with open(data_path+"/seq_labels.p","wb") as file:
            pickle.dump(labels, file)
        

        print("save file preprocessed_data.p successful, execution time {:.2f} s".format(time.time() - t_s))


if __name__ == "__main__":
    data_path = '../data/train'
    total_preprocess(data_path)