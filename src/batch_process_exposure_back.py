#!/usr/bin/env python
# coding=utf-8
import logging
import os
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.grid_search import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import pickle, os, jieba, time, gc, re
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import lightgbm as lgb
import warnings
import math
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
from read_data import *
from dataConfig import dataConfig


def group_data(file1_dir, file2_dir, group_file):

    if os.path.exists(group_file):
        logging.info(group_file + 'already exists')
        return

    dir_list = []
    read_dir = os.listdir(file1_dir)
    for file in read_dir:
        dir_list.append(file)

    write_file_list = ['exposure_group_' + str(i) + '.csv' for i in range(len(dir_list))]

    def handle_one_file(file_1, file_2):
        if not os.path.exists(file_1):
            logging.info(file_1 + ' not exits')
            return
        if os.path.exists(file_2):
            logging.info(file_2 + 'already exists')
            return

        names = ['request_id', 'request_time', 'user_id', 'location_id', 'bidding_Advertising_Information']
        track_log_df = pd.read_csv(file_1, delimiter='\t', \
                                   parse_dates=['request_time'], header=None, names=names, engine='python')
        logging.info('read data success')

        track_log_df['request_time'] = track_log_df['request_time'].astype(int)
        track_log_df['request_time'] = pd.to_datetime(track_log_df['request_time'] + 8 * 3600, unit='s', utc=True)
        track_log_df['request_day'] = track_log_df['request_time'].dt.strftime('%Y-%m-%d')

        logging.info('handle time success')


        # track_log_df['win_ad_count'] = track_log_df['bidding_Advertising_Information'].map(win_ad_count)
        track_log_df['win_ad'] = track_log_df['bidding_Advertising_Information'].map(win_ad)

        track_log_df = (track_log_df.set_index(
            ['request_id', 'location_id', 'user_id', 'request_day', 'bidding_Advertising_Information'])[
                            'win_ad']
                        .str.split('|', expand=True)
                        .stack()
                        .reset_index(level=5, drop=True)
                        .reset_index(name='win_ad'))

        win_ad_columns = ['ad_id', 'bidding_price', 'pctr', 'quality_ecpm', 'totalEcpm']
        for index in range(0, len(win_ad_columns)):
            track_log_df[win_ad_columns[index]] = track_log_df.apply(get_win_ad_attribute, axis=1, args=(index,))

        static_feature_names = ['ad_id', 'create_time', 'ad_acc_id', 'good_id', 'good_class', 'ad_trade_id', 'ad_size']
        ad_static_path = '../../data/total_data/map_ad_static.out'
        ad_static_df = pd.read_csv(ad_static_path, delimiter='\t', \
                                   parse_dates=['create_time'], header=None, names=static_feature_names,
                                   dtype={'ad_id': int, "ad_acc_id": int, \
                                          "good_id": str, "good_class": str, "ad_trade_id": str, 'ad_size': str})
        ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'], unit='s')
        ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')

        del track_log_df['bidding_Advertising_Information']
        del track_log_df['win_ad']
        gc.collect()
        track_log_df['ad_id'] = track_log_df['ad_id'].astype(int , inplace = True )
        track_log_df = pd.merge(track_log_df, ad_static_df, on=['ad_id'], how='left')
        print(track_log_df.shape)
        print(track_log_df.head())
        print(track_log_df.isnull().sum())

        track_log_df.to_csv(file_2, header=True, index=None, sep=';', mode='w')
        del track_log_df
        gc.collect()
        logging.info(file_2 + ' dump success')

    for i, read_file in enumerate(dir_list):
        handle_one_file(file1_dir + read_file, file2_dir + write_file_list[i])

    logging.info('start handle merge group data')
    df1 = pd.read_csv(file2_dir + write_file_list[0], delimiter=';')
    # df1 = df1.set_index(['request_day', 'ad_id'])
    for i in range(1, len(write_file_list)):
        df = pd.read_csv(file2_dir + write_file_list[i], delimiter=';')
        # df = df.set_index(['request_day', 'ad_id'])
        # df1 = df1.add(df, fill_value='0')
        df1 = pd.concat([df1, df], axis=0, ignore_index=True)
        del df
        gc.collect()
    logging.info('merge success')
    df1.to_csv(group_file, index= None, sep='\t')
    print(df1.shape)
    print(df1.isnull().sum())
    logging.info(group_file + ' dump success')

if __name__ == '__main__':
    # group_data('../../data/total_data/track_log_20190419.out../../data/whole_train_data/exposure_group_all.csv')

    # track_log_path = '../../data/total_data/track_log/'
    # exposure_group_data_path = '../../data/chusai_statstic_data/group_data/'
    # exposure_group_all_path =  '../../data/chusai_statstic_data/exposure_group_all.csv'
    data_conf = dataConfig()

    train_track_log_path = data_conf.train_track_log_path
    exposure_group_data_path = data_conf.exposure_group_data_path
    exposure_group_all_path =  data_conf.exposure_group_all_path
    group_data(train_track_log_path, exposure_group_data_path,exposure_group_all_path)
