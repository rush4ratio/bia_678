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
from tqdm import tqdm
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB


logging.basicConfig(format='%(asctime)s %(message)s', 
                    handlers=[logging.FileHandler("pseudo_prep.log"),
                              logging.StreamHandler()], level=logging.DEBUG)



ratings = pd.read_csv('data/training_ratings.csv')

genome = pd.read_csv('data/190mb/genome-scores.csv')

user_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')

user_ratings = user_ratings.reset_index()

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

def process_user(user_id):
    logging.info('processing userId {}'.format(user_id))
    
    user = user_ratings[user_ratings.userId == user_id].drop('userId', axis = 1)
    movie_ratings_per_user_dict = OrderedDict({key: val for key,val in user.iloc[0].items() if pd.notnull(val)})
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
    
    unrated = user.melt()
    unrated = unrated[pd.isnull(unrated.value)]
    
    movie_ids_for_preds = list(unrated.movieId)
    mv_features_preds = get_matrix_features_for_movies(movie_ids_for_preds)
    mv_features_preds = mv_features_preds.reset_index()
    
    
    mv_features_preds['target'] = gnb.predict(mv_features_preds.drop('movieId', axis = 1).values)
    
    mv_features_preds['pseudo_ratings'] = mv_features_preds['target'].astype('float')
    
    original_ratings = user.melt(value_name='ratings')
    new_ratings = pd.merge(original_ratings, mv_features_preds,left_on='movieId', right_on='movieId',how='left')
    new_ratings = new_ratings[['movieId','ratings', 'pseudo_ratings']]
    new_ratings['ratings'] = new_ratings[['ratings', 'pseudo_ratings']].apply(apply_new_ratings, axis =1)

    new_ratings['userId'] = user_id
    
    new_ratings = new_ratings[['userId','movieId','ratings']]
    new_ratings = new_ratings.reset_index(drop=True)
    
    logging.info('Saving user ratings...')
    save_ratings(new_ratings)

def save_ratings(ratings_data):
    ratings_data = ratings_data.dropna(axis=0, how='any')

    if not os.path.isfile('pseudo_training.csv'):
        ratings_data.to_csv('pseudo_training.csv',index=False)
    else:
        ratings_data.to_csv('pseudo_training.csv', mode = 'a', 
                           index=False, header=None)

    
user_ids = list(ratings.userId.unique())
    
for user in tqdm(user_ids):
    process_user(user)
    
    
    
    
    
    
    
    