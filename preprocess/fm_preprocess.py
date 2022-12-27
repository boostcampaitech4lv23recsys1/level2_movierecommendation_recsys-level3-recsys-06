import pandas as pd
import numpy as np
import os

from tqdm import tqdm
import time
from collections import defaultdict
import pickle

from sklearn.preprocessing import LabelEncoder


def make_pickle_files(data_path, user_dict, item_dict, labels):
    data_path = os.path.join(data_path, 'fm_data')
    user_dict_path = os.path.join(data_path, 'fm_user_dict.pkl')
    item_dict_path = os.path.join(data_path, 'fm_item_dict.pkl')
    label_path = os.path.join(data_path, 'fm_labels.pkl')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(user_dict_path,"wb") as file:
        pickle.dump(user_dict,file)

    with open(item_dict_path,"wb") as file:
        pickle.dump(item_dict,file)
    
    if os.path.exists(label_path):
        # Delete the file
        os.remove(label_path)

    with open(label_path,"wb") as file:
        pickle.dump(labels,file)


def release_year_mapping(year,gap):
    if year <= 1930: 
        return 0
    else:
        return (year - 1930) // gap + 1


def make_train_year_data(train_df, year_data, title_data):
    # item release year, categorized_year 만들기
    trainyear_df = pd.merge(train_df, year_data, on=['item'], how='left')

    no_year_item = trainyear_df[trainyear_df.release_year.isnull()].item.unique()
    no_year_df = title_data[title_data['item'].isin(no_year_item)]
    year_na_dict = dict(zip(list(no_year_df.item), list(no_year_df.title.str[-5:-1].astype(np.float64))))
    no_year_fill = trainyear_df[trainyear_df['item'].isin(no_year_item)].index
    
    for index in no_year_fill:
        trainyear_df.at[index,'release_year'] = year_na_dict[trainyear_df.iloc[index]['item']]
        
    trainyear_df['categorized_year_gap5'] = trainyear_df['release_year'].apply(release_year_mapping, args=(5,))
    trainyear_df['categorized_year_gap10'] = trainyear_df['release_year'].apply(release_year_mapping, args=(10,))
    trainyear_df['release_year'] = trainyear_df['release_year'].apply(int)
    
    return trainyear_df


def make_director_total(director_data, popular_director, train_df):
    ## 메인감독 모으기
    director_dict=dict(director_data.groupby('item')['director'].value_counts())

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

    other_movie=list(set(train_df['item'].unique())-set(director_data['item'].unique()))
    other_movie_dict={}

    for i in other_movie:
        other_movie_dict[i]='other_d'
    other_director_frame = pd.DataFrame(other_movie_dict.items(),columns=['item','director'])
    director_total=pd.concat([director_data,other_director_frame],axis=0)
    director_total=director_total.sort_values(by='item')
   
    director_main=director_total.merge(main_dir_frame, how='left',on=['item'])
    director_main['main_director']=director_main['main_director'].fillna('other_d')

    return director_main


def make_writer_total(writer_data, popular_writer, train_df):
    ## 메인작가 모으기
    writer_dict=dict(writer_data.groupby('item')['writer'].value_counts())

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

    other_writer_movie=list(set(train_df['item'].unique())-set(writer_data['item'].unique()))

    other_writer_dict={}
    for i in other_writer_movie:
        other_writer_dict[i]='other_w'
    other_writer_frame = pd.DataFrame(other_writer_dict.items(),columns=['item','writer'])
    writer_total=pd.concat([writer_data,other_writer_frame],axis=0)
    writer_total=writer_total.sort_values(by='item')

    writer_main = writer_total.merge(main_writer_frame, how='left',on=['item'])
    writer_main['main_writer']=writer_main['main_writer'].fillna('other_w')
    
    return writer_main

def load_data(data_path):
    # 전체 학습 데이터
    train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv')) # 전체 학습 데이터
    train_df['date'] = pd.to_datetime(train_df['time'], unit='s')
    train_df = train_df.sort_values(by = ['user','time'], axis = 0)

    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    year_data.rename(columns = {'year':'release_year'},inplace=True)
    
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')

    # 타이틀 데이터
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    # title 중복된 것 처리 => 스필버그 영화의 영화는 더 유명한 우주전쟁, item_id가 64997인 녀석은 안 유명한 우주전쟁
    title_data[title_data['item']==64997].title = 'War of the Worlds_B (2005)'
    title_data.at[1926, 'title'] = 'War of the Worlds_B (2005)'

    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

    return train_df, year_data, writer_data, title_data, genre_data, director_data


def fm_preprocess(data_path):
    user_dict_path = os.path.join(data_path, 'fm_data/fm_user_dict.pkl')
    item_dict_path = os.path.join(data_path, 'fm_data/fm_item_dict.pkl')

    if os.path.exists(user_dict_path) and os.path.exists(item_dict_path):
        with open(user_dict_path, "rb") as file:
            user_dict = file

        with open(item_dict_path, "rb") as file:
            item_dict = file

        print("user dictionary and item dictionary alreay exist")

    else:
        st = time.time()
        train_df, year_data, writer_data, title_data, genre_data, director_data = load_data(data_path)  

        # user favorite_genre, maniatic 만들기
        tem = pd.get_dummies(genre_data) 
        genre_data_one = tem.groupby('item').sum().reset_index(); genre_data_one

        for col in genre_data_one.columns[1:] :
            genre_data_one.rename(columns = {col : col[6:]}, inplace = True)
             
        useritem_genre = pd.merge(train_df.iloc[:,:2], genre_data_one, on=['item'], how='left'); useritem_genre
        useritem_count = useritem_genre.groupby('user')['item'].count().reset_index();useritem_count
        user_genre = useritem_genre.groupby('user').sum().reset_index();user_genre
        user_genre['favorite_genre'] = user_genre.iloc[:, 2:].idxmax(axis=1)
        user_genre['maniatic'] = user_genre.iloc[:, 2:].max(axis=1) / useritem_count['item']
        user_genre[['user','favorite_genre','maniatic']]

        user_df = user_genre[['user', 'favorite_genre', 'maniatic']]
        
        #item release year 피쳐를 만들고 범주형으로 바꿔줌
        trainyear_df = make_train_year_data(train_df, year_data, title_data)

        director_concat_data = trainyear_df.merge(director_data, how='left',on=['item'])
        writer_concat_data = trainyear_df.merge(writer_data,how='left',on=['item'])
        popular_director=director_concat_data['director'].value_counts()
        popular_writer=writer_concat_data['writer'].value_counts()

        # 결측치를 other로 채워줌
        director_main = make_director_total(director_data, popular_director, train_df)
        writer_main = make_writer_total(writer_data, popular_writer, train_df)

        director_main = director_main.merge(title_data, how='left', on=['item'])
        writer_main = writer_main.merge(title_data, how='left', on=['item'])

        # 대표 작가가 없는 항목에 대표 감독을 넣어줌
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

        dir_frame = pd.DataFrame(main_directors.items(),columns=['item','main_director'])
        writer_frame = pd.DataFrame(main_writers.items(),columns=['item','main_writer'])

        trainyear_df = trainyear_df.merge(user_genre[['user','favorite_genre','maniatic']], how='left', on=['user'])
        trainyear_df = trainyear_df.merge(title_data, how='left', on=['item'])

        train_df_dropped = trainyear_df[['user', 'favorite_genre', 'maniatic']]
        item_year_df = trainyear_df[['item', 'title', 'release_year', 'categorized_year_gap5', 'categorized_year_gap10']]
        item_year_df = item_year_df.drop_duplicates().sort_values(by='item')

        ##user의 정보를 담은 딕셔너리를 만들 df : user_essence
        user_essence = train_df_dropped.drop_duplicates()

        print(f"labelling start")
        ## Label Encoding
        # 감독/작가 라벨
        dw_label=LabelEncoder()
        dw_label.fit(pd.concat([director_main['director'],writer_main['writer']],axis=0))

        # 유저 라벨
        user_label=LabelEncoder()
        user_label.fit(train_df['user'])

        # 장르 라벨
        genre_label=LabelEncoder()
        genre_label.fit(genre_data['genre'])

        # 영화 라벨
        item_label=LabelEncoder()
        item_label.fit(train_df['item'])

        # 연도 라벨은 이미 되어 있어서 아래의 코드로 라벨링해주기만 하면 됨
        # trainyear_df['release_year'] = trainyear_df['release_year'].apply(release_year_mapping, args=(1,))

        # 제목 라벨
        title_label = LabelEncoder()
        title_label.fit(title_data['title'])

        labels = [dw_label, user_label, genre_label, item_label, title_label]
        """
        필요한 다섯 df에 모두 라벨링하기
        """

        director_main['item'] = item_label.transform(director_main['item'])
        director_main['director'] = dw_label.transform(director_main['director'])
        director_main['main_director'] = dw_label.transform(director_main['main_director'])
        director_main['title'] = title_label.transform(director_main['title'])

        writer_main['item'] = item_label.transform(writer_main['item'])
        writer_main['writer'] = dw_label.transform(writer_main['writer'])
        writer_main['main_writer'] = dw_label.transform(writer_main['main_writer'])
        writer_main['title'] = title_label.transform(writer_main['title'])

        user_essence['user'] = user_label.transform(user_essence['user'])
        user_essence['favorite_genre'] = genre_label.transform(user_essence['favorite_genre'])

        trainyear_df['user'] = user_label.transform(trainyear_df['user'])
        trainyear_df['item'] = item_label.transform(trainyear_df['item'])
        trainyear_df['release_year'] = trainyear_df['release_year'].apply(release_year_mapping, args=(1,))
        trainyear_df['favorite_genre'] = genre_label.transform(trainyear_df['favorite_genre'])

        item_year_df['item'] = item_label.transform(item_year_df['item'])
        item_year_df['release_year'] = item_year_df['release_year'].apply(release_year_mapping, args=(1,))
        item_year_df['title'] = title_label.transform(item_year_df['title'])

        genre_data['item'] = item_label.transform(genre_data['item'])
        genre_data['genre'] = genre_label.transform(genre_data['genre'])
       
        print(f"labelling finished")
        trainyear_df = trainyear_df[['user','favorite_genre','maniatic']]


        """### 사용하는 데이터프레임들
        - user_essence
        - writer_main
        - director_main
        - trainyear_df
        - item_year_df
        """

        n_user = len(user_essence)
        user_dict = {}
        print("user_dict generating")
        for i in tqdm(range(n_user)):
            temp = user_essence.iloc[i]
            user_dict[temp.user] = {'favorite_genre': [temp.favorite_genre], 'maniatic': [temp.maniatic]}
        
        print("item_dict generating")
        item_writer = dict(writer_main.groupby('item')['writer'].value_counts()).keys()
        writer_dict= defaultdict(list)
       
        for item, writer in item_writer:
            writer_dict[item].append(writer)

        item_director = dict(director_main.groupby('item')['director'].value_counts()).keys()
        director_dict= defaultdict(list)

        for item, director in item_director:
            director_dict[item].append(director)

        item_main_director = dict(director_main.groupby('item')['main_director'].value_counts()).keys()
        main_director_dict = defaultdict(list)

        for item, main_director in item_main_director:
            main_director_dict[item].append(main_director)

        item_main_writer = dict(writer_main.groupby('item')['main_writer'].value_counts()).keys()
        main_writer_dict = defaultdict(list)

        for item, main_writer in item_main_writer:
            main_writer_dict[item].append(main_writer)

        item_genre = dict(genre_data.groupby('item')['genre'].value_counts()).keys()
        genre_item_dict = defaultdict(list)

        for item, genre in item_genre:
            genre_item_dict[item].append(genre)
        
        n_item = len(train_df.item.unique())
        item_dict = defaultdict(dict)
        for i in tqdm(range(n_item)):
            temp = item_year_df.iloc[i]
            item_dict[i]['release_year'] = [temp.release_year]
            item_dict[i]['categorized_year_gap5'] = [temp.categorized_year_gap5]
            item_dict[i]['categorized_year_gap10'] = [temp.categorized_year_gap10]
            item_dict[i]['title'] = [temp.title]
            item_dict[i]['director'] = director_dict[i]
            item_dict[i]['main_director'] = main_director_dict[i]
            item_dict[i]['writer'] = writer_dict[i]
            item_dict[i]['main_writer'] = main_writer_dict[i]
            item_dict[i]['genre'] = genre_item_dict[i]
        
        make_pickle_files(data_path, user_dict, item_dict, labels)

        print("complete fm preprocessing, execution time {:.2f} s".format(time.time() - st))


if __name__ == "__main__":
    data_path = '../data/train'
    fm_preprocess(data_path)