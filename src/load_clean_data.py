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
from datetime import datetime, timedelta
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
from utils import *
from sklearn.externals import joblib

warnings.filterwarnings("ignore")
from modelPath import modelPath
from dataConfig import  dataConfig
from read_data import *

model_path = modelPath()
# file_path = data_conf.file_path
logging.basicConfig(level=logging.INFO)

my_load_clean_with_dynamic_data_path = model_path.ad_id_train_path
# my_load_clean_with_dynamic_data_path = '../data/try/train_data.csv'
test_sample_bid_path = model_path.ad_id_test_path
ad_op_mid_path = model_path.ad_op_mid_simple_path
# ad_op_mid_path = '../data/total_data/process_simple_ad_data/ad_op_mid_simple.txt'
# my_load_clean_with_dynamic_data_path = model_path
def filter(x):
    if (x['ad_id_request_num'] < 100 and x['exp_num'] - 0 == 0):
        return 1
    else:
        return 0
    # 459154

def my_try_secondF_load_clean_with_dynamic_data():
    # my_load_clean_with_dynamic_data_path = '../data/total_data/process_train_data/train_data.csv'
    # my_load_clean_with_dynamic_data_path = '../data/diff_ecpm/sample_train_data.csv'
    # my_load_clean_with_dynamic_data_path = '../data/diff_ecpm/train_data.csv'
    # my_load_clean_with_dynamic_data_path =     '../data/try/sample_train_data.csv'

    simple_train_data = pd.read_csv(my_load_clean_with_dynamic_data_path, delimiter='\t')
    simple_train_data.dropna(inplace = True )
###########  train
    static_feature_names = ['ad_id', 'create_time', 'ad_acc_id', 'good_id', 'good_class', 'ad_trade_id', 'ad_size']
    ad_static_path = '../data/total_data/map_ad_static.out'
    ad_static_df = pd.read_csv(ad_static_path, delimiter='\t', \
                               parse_dates=['create_time'], header=None, names=static_feature_names,
                               dtype={'ad_id': int, "ad_acc_id": int, \
                                      "good_id": str, "good_class": str, "ad_trade_id": str, 'ad_size': str})
    # ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'], unit='s')
    # ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')
    ad_static_df['create_time'] = ad_static_df['create_time'].astype(int)
    ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'] + 8 * 3600, unit='s', utc=True)
    ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')

    # ad_op_mid_path = '../data/total_data/process_simple_ad_data/ad_op_mid_simple.txt'
    new_op_df = pd.read_csv(ad_op_mid_path, delimiter='\t')
    new_op_df['ad_id'] = new_op_df['ad_id'].astype(int)

    # left_ad_static_df_right_ad_op_df = ad_static_df.merge(new_op_df, on='ad_id', how='left')
    left_ad_op_right_ad_static_df_df = new_op_df.merge(ad_static_df, on='ad_id', how='left')

    # ad_accurately_path = '../data/testFuSai/process_ad_data/ad_static_dynamic_merge_accurately.csv'
    # left_ad_op_right_ad_static_df_df = pd.read_csv(ad_accurately_path, delimiter='\t')
    # drop_columns  = [ 'valid_start_time' , 'valid_end_time' ,'request_day',
    #                                        'exp_num','res' ]
    # for col in drop_columns:
    #      del left_ad_op_right_ad_static_df_df[ col ]
    # gc.collect()

    simple_train_data['ad_id'] = simple_train_data['ad_id'].astype(int)
    with_dynamic_train_df = pd.merge(left_ad_op_right_ad_static_df_df, simple_train_data, on=['ad_id'],
                                             how='left')
    # with_dynamic_whole_train_data = with_dynamic_whole_train_data.dropna()
    with_dynamic_train_df.dropna(inplace=True)
    logging.info('*****  with_dynamic_train_df ***********')
    print(with_dynamic_train_df.isnull().sum())
    logging.info('*****  with_dynamic_train_df ***********')
############## test
    # test_sample_bid_path = '../data/total_data/test_sample_bid.out'
    # names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    # test_sampleDF = pd.read_csv(test_sample_bid_path, delimiter='\t', \
    #                                       header=None, names=names,dtype={"sample_id": int
    #         , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int })
    #
    # test_sampleDF = pd.merge(test_sampleDF, ad_static_df, on=['ad_id'],
    #                                          how='left')
    # test_sample_bid_path = '../data/total_data/test_data/guize/simple_whole_test.txt'

    # test_sample_bid_path = '../data/total_data/test_data/new_est/back_with_num_of_opponents_simple_whole_test.txt'

    test_sampleDF = pd.read_csv(test_sample_bid_path)

    logging.info('*****  test_sampleDF ***********')
    print(test_sampleDF.isnull().sum())
    logging.info('*****  test_sampleDF ***********')
    try:
        train = with_dynamic_train_df[with_dynamic_train_df['request_day'] < '2019-04-22']
    except:
        print(with_dynamic_train_df.dtypes)
        print(with_dynamic_train_df.head())
    val = with_dynamic_train_df[with_dynamic_train_df['request_day'] == '2019-04-22']

    train['flag'] = 1
    val['flag'] = 2
    test_sampleDF['flag'] = 3

    # train = train.head(100)
    # val = val.head(10)
    # test_sampleDF= test_sampleDF.head(10)

    data = pd.concat([train, val, test_sampleDF], axis=0, ignore_index=True)

    print('****************data******************')
    data['filter'] = data.apply(filter, axis=1)
    data = data[data['filter'] == 0]
    print(data.shape)
    print(data.isnull().sum())
    convert_columns = ['ad_acc_id', 'ad_size', 'ad_trade_id', 'good_class', 'good_id']
    for col in convert_columns:
        data[col] = data[col].astype(int, inplace = True)
    print(data.dtypes)
    # print(data.head())
    return data

def my_load_clean_with_dynamic_data():
    # my_load_clean_with_dynamic_data_path = '../data/total_data/process_train_data/train_data.csv'
    # my_load_clean_with_dynamic_data_path = '../data/diff_ecpm/sample_train_data.csv'

    # my_load_clean_with_dynamic_data_path = '../data/diff_ecpm/train_data.csv'
    simple_train_data = pd.read_csv(my_load_clean_with_dynamic_data_path, delimiter='\t')
    simple_train_data.dropna(inplace = True )
###########  train
    static_feature_names = ['ad_id', 'create_time', 'ad_acc_id', 'good_id', 'good_class', 'ad_trade_id', 'ad_size']
    ad_static_path = '../data/total_data/map_ad_static.out'
    ad_static_df = pd.read_csv(ad_static_path, delimiter='\t', \
                               parse_dates=['create_time'], header=None, names=static_feature_names,
                               dtype={'ad_id': int, "ad_acc_id": int, \
                                      "good_id": str, "good_class": str, "ad_trade_id": str, 'ad_size': str})
    # ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'], unit='s')
    # ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')
    ad_static_df['create_time'] = ad_static_df['create_time'].astype(int)
    ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'] + 8 * 3600, unit='s', utc=True)
    ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')


    new_op_df = pd.read_csv(ad_op_mid_path, delimiter='\t')
    new_op_df['ad_id'] = new_op_df['ad_id'].astype(int)

    # left_ad_static_df_right_ad_op_df = ad_static_df.merge(new_op_df, on='ad_id', how='left')
    left_ad_op_right_ad_static_df_df = new_op_df.merge(ad_static_df, on='ad_id', how='left')

    # ad_accurately_path = '../data/testFuSai/process_ad_data/ad_static_dynamic_merge_accurately.csv'
    # left_ad_op_right_ad_static_df_df = pd.read_csv(ad_accurately_path, delimiter='\t')
    # drop_columns  = [ 'valid_start_time' , 'valid_end_time' ,'request_day',
    #                                        'exp_num','res' ]
    # for col in drop_columns:
    #      del left_ad_op_right_ad_static_df_df[ col ]
    # gc.collect()

    simple_train_data['ad_id'] = simple_train_data['ad_id'].astype(int)
    with_dynamic_train_df = pd.merge(left_ad_op_right_ad_static_df_df, simple_train_data, on=['ad_id'],
                                             how='left')
    # with_dynamic_whole_train_data = with_dynamic_whole_train_data.dropna()
    with_dynamic_train_df.dropna(inplace=True)
    logging.info('*****  with_dynamic_train_df ***********')
    print(with_dynamic_train_df.isnull().sum())
    logging.info('*****  with_dynamic_train_df ***********')
############## test
    # test_sample_bid_path = '../data/total_data/test_sample_bid.out'
    # names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    # test_sampleDF = pd.read_csv(test_sample_bid_path, delimiter='\t', \
    #                                       header=None, names=names,dtype={"sample_id": int
    #         , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int })
    #
    # test_sampleDF = pd.merge(test_sampleDF, ad_static_df, on=['ad_id'],
    #                                          how='left')
    # test_sample_bid_path = '../data/total_data/test_data/guize/simple_whole_test.txt'
    test_sampleDF = pd.read_csv(test_sample_bid_path)

    logging.info('*****  test_sampleDF ***********')
    print(test_sampleDF.isnull().sum())
    logging.info('*****  test_sampleDF ***********')
    try:
        train = with_dynamic_train_df[with_dynamic_train_df['request_day'] < '2019-04-22']
    except:
        print(with_dynamic_train_df.dtypes)
        print(with_dynamic_train_df.head())
    val = with_dynamic_train_df[with_dynamic_train_df['request_day'] == '2019-04-22']

    train['flag'] = 1
    val['flag'] = 2
    test_sampleDF['flag'] = 3

    # train = train.head(100)
    # val = val.head(10)
    # test_sampleDF= test_sampleDF.head(10)

    data = pd.concat([train, val, test_sampleDF], axis=0, ignore_index=True)

    print('****************data******************')
    data['filter'] = data.apply(filter, axis=1)
    data = data[data['filter'] == 0]
    print(data.shape)
    print(data.isnull().sum())
    convert_columns = ['ad_acc_id', 'ad_size', 'ad_trade_id', 'good_class', 'good_id']
    for col in convert_columns:
        data[col] = data[col].astype(int, inplace = True)
    print(data.dtypes)
    # print(data.head())
    return data


