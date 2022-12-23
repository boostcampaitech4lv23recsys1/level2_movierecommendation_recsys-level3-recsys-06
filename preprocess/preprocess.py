import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

path = "../../data/train/"
director = pd.read_csv(path+"directors.tsv",sep='\t')
genre = pd.read_csv(path+"genres.tsv",sep='\t')
title = pd.read_csv(path+"titles.tsv",sep='\t')
train_rating = pd.read_csv(path+"train_ratings.csv",sep=',')
writer = pd.read_csv(path+"writers.tsv",sep='\t')
year = pd.read_csv(path+"years.tsv",sep='\t')

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
    director_dict=dict(director.groupby('item').value_counts())
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

def item_labeling():
    item_label = LabelEncoder()
    total_label = item_label.fit(train_rating['item'])
    return total_label 

def director_and_writer_labeling():
    director_main = director_preprocess()
    writer_main = writer_preprocess()
    writer_main = writer_fillna()
    le_director_and_writer=LabelEncoder()
    item_label = item_labeling()
    total_label = le_director_and_writer.fit(pd.concat([director_main['director'],writer_main['writer']],axis=0))
    director_main['item_label'] = item_label.transform(director_main['item']) 
    director_main['director_label'] = total_label.transform(director_main['director'])
    director_main['main_director_label'] = total_label.transform(director_main['main_director'])
    writer_main['item_label'] = item_label.transform(writer_main['item']) 
    writer_main['writer_label'] = total_label.transform(writer_main['writer'])
    writer_main['main_writer_label'] = total_label.transform(writer_main['main_writer'])
    # with open("director_and_writer_label.pkl","wb") as file:
    #     pickle.dump(total_label,file)
    return director_main,writer_main

def director_and_writer_preprocess():
    director_file,writer_file=director_and_writer_labeling()
    director_list = {}
    for i in range(len(director_main)):
        tmp = director_main.iloc[i]
        director_list[tmp['item_label']]=director_list.get(tmp['item_label'],[])+[tmp['director_label']]
    writer_list = {}
    for i in range(len(writer_main)):
        tmp = writer_main.iloc[i]
        writer_list[tmp['item_label']]=writer_list.get(tmp['item_label'],[])+[tmp['writer_label']]
    main_director_list = {}
    for i in range(len(dir_frame)):
        tmp = dir_frame.iloc[i]
        main_director_list[tmp['item_label']]=[tmp['main_director_label']]
    main_writer_list = {}
    for i in range(len(writer_frame)):
        tmp = writer_frame.iloc[i]
        main_writer_list[tmp['item_label']]=[tmp['main_writer_label']]
    return director_list,writer_list,main_director_list,main_writer_list

def title_preprocess():
    # title[title['title']=='War of the Worlds (2005)']
    # director[director['item']==34048]  # 스필버그 영화 => 더 유명한 우주전쟁
    # item_id가 64997인 녀석은 안 유명한 우주전쟁
    title[title['item']==64997].title = 'War of the Worlds_B (2005)'
    title.at[1926, 'title'] = 'War of the Worlds_B (2005)'
    item_label = item_labeling()
    title['item_label'] = item_label.transform(title['item'])
    title_le = LabelEncoder()
    title_label=title_le.fit(title['title'])
    title['title_label'] = title_le.transform(title['title'])
    title_list={}
    for i in range(len(title)):
        tmp = title.iloc[i]
        title_list[tmp['item_label']]=[tmp['labeled_title']]
    return title_list

