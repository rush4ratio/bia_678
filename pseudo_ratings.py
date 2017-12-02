#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:43:41 2017

@author: rush
"""
import pandas as pd
import os
import numpy as np
import logging

import multiprocessing as mp

import time
from tqdm import tqdm
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB


logging.basicConfig(format='%(asctime)s %(message)s', 
                    handlers=[logging.FileHandler("pseudo_prep-{}.log".format(time.time())),
                              logging.StreamHandler()], level=logging.DEBUG)

np.random.seed(4)

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    logging.debug("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # logging.debug current column type
            logging.debug("******************************")
            logging.debug("Column: ",col)
            logging.debug("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # logging.debug new column type
            logging.debug("dtype after: ",props[col].dtype)
            logging.debug("******************************")
    
    # logging.debug final result
    logging.debug("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    logging.debug("Memory usage is: ",mem_usg," MB")
    logging.debug("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

ratings = pd.read_csv('data/train_acc.csv')


ratings, NAs = reduce_mem_usage(ratings)

genome = pd.read_csv('data/genome-scores.csv')

genome, NAs = reduce_mem_usage(genome)

genome_movie_ids = set(genome.movieId.unique())

def get_matrix_features_for_movies(list_of_movie_ids):
    list_of_movie_ids = set(list_of_movie_ids)
    mv_id = genome[genome.movieId.apply(lambda x: x in list_of_movie_ids)].copy()
    mv_id_pivot = mv_id.pivot(index='movieId', columns='tagId', values='relevance')
    return mv_id_pivot

def apply_new_ratings(x):
    if pd.isnull(x[0]):
        return x[1]
    elif pd.notnull(x[0]):
        return x[0]

def process_user(user_id, user_ratings, movie_id_lst):
    logging.info('processing userId {}'.format(user_id))
    
    movie_ratings_per_user_dict = OrderedDict({key: val for key,val in user_ratings.iloc[0].items() if pd.notnull(val)})
    movie_ids = list(movie_ratings_per_user_dict.keys())
    mv_features = get_matrix_features_for_movies(movie_ids)
    mv_features = mv_features.reset_index()

    ratings_df = pd.DataFrame.from_dict(movie_ratings_per_user_dict, orient='index').reset_index()
    ratings_df.columns = ['movieId', 'rating']
    
    data = pd.merge(mv_features, ratings_df, left_on='movieId', 
                    right_on='movieId', how='inner').drop('movieId', axis = 1)
    
    data['target'] = data.rating.apply(lambda x: str(x))
    
    logging.info('Training labels...')
    gnb = GaussianNB()
    gnb.fit(data.drop(['rating','target'], axis = 1).values,data.target.values)
    
    movie_ids_for_preds = list((movie_id_lst - set(movie_ids)) & genome_movie_ids)
    
    movie_size = len(movie_ids_for_preds)
    
    new_movie_ids = set()
    while len(new_movie_ids) < 100:
        val = np.random.randint(movie_size)  
        try:
            new_movie_ids.add( movie_ids_for_preds[val] )
        except IndexError:
           pass        
        
    mv_features_preds = get_matrix_features_for_movies(list(new_movie_ids))
    mv_features_preds = mv_features_preds.reset_index()
    
    mv_features_preds['target'] = gnb.predict(mv_features_preds.drop('movieId', axis = 1).values)
    
    mv_features_preds['pseudo_ratings'] = mv_features_preds['target'].astype('float')
    
    original_ratings = user_ratings.melt(value_name='ratings')
    
    new_ratings = pd.merge(original_ratings, mv_features_preds,left_on='movieId', right_on='movieId',how='outer')
    new_ratings = new_ratings[['movieId','ratings', 'pseudo_ratings']]
    
    
    new_ratings['ratings'] = new_ratings[['ratings', 'pseudo_ratings']].apply(apply_new_ratings, axis =1)

    new_ratings['userId'] = user_id
    
    new_ratings = new_ratings[['userId','movieId','ratings']]
    #new_ratings = new_ratings.reset_index(drop=True)
    
    return new_ratings
    

def save_ratings(file_name, ratings_data):
    ratings_data.to_csv('data/users/'+ file_name,index=False)
    
    
def chunks(L,n):
    for i in range(0, len(L),n):
        yield L[i:i+n]

    
def process_sub_list_of_users(sub_list):   
    movie_id_lst = set(ratings.movieId.unique())
    
    for user in tqdm(sub_list):
        ratings_user = ratings[ratings.userId == user]
        user_ratings = ratings_user.pivot(index='userId', columns='movieId', values='rating')
        user_ratings = user_ratings.reset_index()
        
        user_ratings.drop('userId', axis = 1, inplace=True)
        
        the_new_ratings = process_user(user, user_ratings, movie_id_lst)
        
        logging.info('Saving user ratings...')
        save_ratings('user_id_{}.csv'.format(user), the_new_ratings)
        
 

user_id_list = list(ratings.userId.unique())

# user_id_lis			t = user_id_list[30000:]

logging.info("user id length: {}".format(len(user_id_list)))
  
generate_user_ids = chunks(user_id_list,1600)

processes = [mp.Process(target=process_sub_list_of_users, args=(x,)) for x in generate_user_ids]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()


    
    