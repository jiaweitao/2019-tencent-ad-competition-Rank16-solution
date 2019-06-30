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
from dataConfig import dataConfig
from utils import *
from read_data import *
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.INFO)

data_conf = dataConfig()

test_track_log_path = data_conf.test_track_log_path
final_select_test_request_path = data_conf.final_select_test_request_path
test_sample_bid_path = data_conf.test_sample_bid_path

with_dynamic_ad_id_whole_test_path = data_conf.with_dynamic_ad_id_whole_test_path
simple_ad_id_whole_test_path = data_conf.simple_ad_id_whole_test_path
ad_static_path = data_conf.ad_static_path

def extract_compete_queue_info():
    test_track_log_df, final_select_test_request_df, test_sample_bid_path_df = read_test_data(
        test_track_log_path,
        final_select_test_request_path,
        test_sample_bid_path)

    print('test_track_log_df   ', test_track_log_df.shape)
    print('final_select_test_request_df    ', final_select_test_request_df.shape)
    print('test_sample_bid_path_df   ', test_sample_bid_path_df.shape)

    test_track_log_df['queue_length'] = test_track_log_df['bidding_Advertising_Information'].map(
        lambda x: len(x.split(';')))

    del test_track_log_df['bidding_Advertising_Information']
    gc.collect()

    def get_request_id_and_location_id(x, index):
        return x['request_set'].split(',')[index]

    final_select_test_request_df = (final_select_test_request_df.set_index(['ad_id'])['request_set']
                                    .str.split('|', expand=True)
                                    .stack()
                                    .reset_index(level=1, drop=True)
                                    .reset_index(name='request_set'))
    print(  final_select_test_request_df.head()  )

    columns = ['request_id', 'location_id']
    for index in range(0, len(columns)):
        final_select_test_request_df[columns[index]] = final_select_test_request_df.apply(
            get_request_id_and_location_id, axis=1, args=(index,))

    length = final_select_test_request_df['request_set'].str.count('|').sum()
    print('ls shape:  ', length)

    del final_select_test_request_df['request_set']
    gc.collect()

    convert_dtype_columns = ['request_id', 'location_id']
    final_select_test_request_df[convert_dtype_columns] = final_select_test_request_df[convert_dtype_columns].astype(
        int, inplace=True)

    print( 'dtypes:  ',  final_select_test_request_df.dtypes )
    print( 'dtypes:  ',  test_track_log_df.dtypes)

    final_select_test_request_df_with_track_df = pd.merge(final_select_test_request_df, test_track_log_df,
                                                          on=['request_id', 'location_id'], how='left')

    print(final_select_test_request_df_with_track_df.head())
    print(final_select_test_request_df_with_track_df.isnull().sum())

    final_select_test_request_df_with_track_df.to_csv(with_dynamic_ad_id_whole_test_path, index=None)
    return final_select_test_request_df_with_track_df

def get_model_test_data():
    # saved_final_select_test_request_df = '../../data/total_data/test_data/sample_final_select_test_request_df'
    start_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    print('start time:        ', start_time)
    # if os.path.exists(simple_whole_test_path):
    #     logging.info("***begin*******read_saved_final_select_test_request_df begin*******")
    #     final_select_test_request_df = pd.read_csv(simple_whole_test_path,
    #                 dtype={"request_id": str, 'user_id': str} )
    #     logging.info("***end*******read_saved_final_select_test_request_df end*******")
    #     print('final_select_test_request_df: ', final_select_test_request_df.shape)
    #     print(final_select_test_request_df.dtypes)
    # else:
    if os.path.exists(with_dynamic_ad_id_whole_test_path):
        # with_dynamic_whole_test_path = '../../data/total_data/test_data/final_select_test_request_df_with_track_df'
        # with_path = '../../data/total_data/test_data/smaple_final_select_test_request_df_with_track_df'
        final_select_test_request_df_with_track_df = pd.read_csv(with_dynamic_ad_id_whole_test_path)
    else:
        start_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
        print('start time:        ', start_time)

        final_select_test_request_df_with_track_df = extract_compete_queue_info()

        end_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
        print('start time:        ', end_time)


    print(final_select_test_request_df_with_track_df.shape)
    print(final_select_test_request_df_with_track_df.dtypes)
    print(final_select_test_request_df_with_track_df.isnull().sum())

    with_num_of_opponents_track_df = extract_test_ad_id_track_log_num_of_opponents(final_select_test_request_df_with_track_df)

    ad_static_df = read_ad_static_data(ad_static_path='../../data/total_data/map_ad_static.out')
    test_sample_bid_path = '../../data/total_data/BTest/Btest_sample_bid.out'
    # test_sample_bid_path_df = read_test_sample_bid_path_df(test_sample_bid_path)

    test_track_log_df, ad_id_one_day_request_num_df, test_sample_bid_path_df = read_test_data(
        test_track_log_path,
        final_select_test_request_path,
        test_sample_bid_path)

    def get_request_id_and_location_id(x):
        return len(x.split('|'))

    ad_id_one_day_request_num_df['ad_id_request_num'] = ad_id_one_day_request_num_df['request_set'].map(
        get_request_id_and_location_id)

    del ad_id_one_day_request_num_df['request_set']
    gc.collect()


    ad_id_one_day_request_num_df['ad_id'] = ad_id_one_day_request_num_df['ad_id'].astype(int, inplace=True)
    ad_static_df['ad_id'] = ad_static_df['ad_id'].astype(int, inplace=True)
    test_sample_bid_path_df['ad_id'] = test_sample_bid_path_df['ad_id'].astype(int, inplace=True)
    with_num_of_opponents_track_df['ad_id'] = with_num_of_opponents_track_df['ad_id'].astype(int, inplace=True)


    simple_whole_test_data = pd.merge(test_sample_bid_path_df, ad_static_df, on=['ad_id'],
                                      how='left')

    simple_whole_test_data.sort_values(by=['ad_id'], inplace=True)
    print(simple_whole_test_data.shape)
    print(simple_whole_test_data.head(10))

    print(with_num_of_opponents_track_df.shape)
    with_num_of_opponents_track_df.sort_values(by=['ad_id'], inplace=True)
    print(with_num_of_opponents_track_df.head(10))


    simple_whole_test_data = pd.merge(simple_whole_test_data, with_num_of_opponents_track_df, on=['ad_id'],
                                      how='left')


    simple_whole_test_data = pd.merge(simple_whole_test_data, ad_id_one_day_request_num_df, on=['ad_id'],
                                      how='left')

    print(simple_whole_test_data.head(50))
    simple_whole_test_data.to_csv(simple_ad_id_whole_test_path, index=None)

    # end_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    # print('end time:        ', end_time)

if __name__ == '__main__':
    get_model_test_data()


