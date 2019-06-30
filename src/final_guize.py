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

import logging
from modelPath import *
from load_clean_data import *
logging.basicConfig(level=logging.INFO)


# path =  '../../data/try_location_id/train_data.csv'
# test_sample_bid_path = '../../data/try_location_id/test_data/simple_whole_test.txt'

guize_path = modelPath()
path = guize_path.path
submissionfilename = guize_path.submissionfilename
simple_whole_test = guize_path.simple_whole_test


def adjust_new_mom_exp_num(test):
    test['exp_num'] = test['exp_num'].map(lambda x: max(x, 0))
    test['sample_id'] = test['sample_id'].astype(int)
    test_sampleDF = test

    test.sort_values(by=["ad_id", "bid"], inplace=True)
    temp_test_sampleDF = test.copy()
    bid = temp_test_sampleDF.set_index('sample_id')[['ad_id', 'bid']].groupby('ad_id')['bid'].apply(
        lambda row: pd.Series(dict(zip(row.index, row.rank() * 0.0001)))).reset_index()

    test_sampleDF['exp_num'] = test_sampleDF['exp_num'].apply(lambda x: round(x, 0))
    test_sampleDF['adjusted_exp_num'] = test_sampleDF['exp_num'].values + bid['bid'].values
    return test_sampleDF

def zuiqiangguize_location_id_per_times_winning_rate():
    # path = '../../data/total_data/process_train_data/train_data.csv'
    data = pd.read_csv(path ,  delimiter= '\t')
    # print(data.shape)


    group_df = data.groupby(['ad_id','location_id'])\
        ['exp_num', 'num_of_opponents'].\
        agg({'exp_num':'sum',
             'num_of_opponents':'sum'}).reset_index()
    group_df['per_times_winning_rate'] = group_df['exp_num'] / group_df['num_of_opponents']
    del group_df['num_of_opponents']
    del group_df['exp_num']
    gc.collect()
    # print('data聚合')
    # print(group_df.head())
    # print(group_df.shape)
    # test_sample_bid_path = '../../data/total_data/test_data/guize/simple_whole_test.txt'

    # test_sample_bid_path = '../../data/try_location_id/test_data/simple_whole_test.txt'
    test_sampleDF = pd.read_csv(simple_whole_test , usecols=['sample_id',
                                                    'ad_id', 'location_id','bid','num_of_opponents'])
    # print('test_sampleDF')
    test_sampleDF.sort_values(by=['sample_id','ad_id', 'bid'], inplace=True)
    # print(test_sampleDF['sample_id'].value_counts())
    # print(test_sampleDF.head(20))
    # print(test_sampleDF.shape)

    group_df['ad_id'] = group_df['ad_id'].astype(int ,inplace = True )
    test_sampleDF['ad_id'] = test_sampleDF['ad_id'].astype(int, inplace=True)
    group_df['location_id'] = group_df['location_id'].astype(int, inplace=True)
    test_sampleDF['location_id'] = test_sampleDF['location_id'].astype(int, inplace=True)


    test_sampleDF = pd.merge(test_sampleDF, group_df, on = ['ad_id','location_id'], how= 'left')

    test_sampleDF.fillna(0, inplace=True)
    test_sampleDF['exp_num'] = test_sampleDF['per_times_winning_rate'] * test_sampleDF['num_of_opponents']
    # print(test_sampleDF.isnull().sum())
    test_sampleDF.sort_values(by=['ad_id','bid'], inplace= True )
    # print(test_sampleDF.head())
    # print(test_sampleDF.shape)


    test_sampleDF = test_sampleDF.groupby(['sample_id','ad_id', 'bid']) \
        ['exp_num']. \
        agg({'exp_num': 'sum'}).reset_index()

    # print(test_sampleDF.shape)
    # print(test_sampleDF.head())

    # group_df = test_sampleDF.groupby(['ad_id', 'bid']) \
    #     ['exp_num']. \
    #     agg({'exp_num': 'sum'}).reset_index()

    # print(group_df.shape)
    # print(group_df.head())
    # print('mean:  ', np.mean(group_df['exp_num']))
    # print('sum:  ', np.sum(group_df['exp_num']))
    # print('min:  ', np.min(group_df['exp_num']))
    # print('max:  ', np.max(group_df['exp_num']))


    # temp_test_sampleDF = test_sampleDF.copy()
    # bid = temp_test_sampleDF.set_index('sample_id')[['ad_id', 'bid']].groupby('ad_id')['bid'].apply(
    #     lambda row: pd.Series(dict(zip(row.index, row.rank() * 0.0001)))).reset_index()
    # test_sampleDF['exp_num'] = test_sampleDF['exp_num'].apply(lambda x: round(x, 0))
    # test_sampleDF['adjusted_exp_num'] = test_sampleDF['exp_num'].values + bid['bid'].values
    # print('test_sampleDF: ')
    # print(test_sampleDF[['sample_id', 'ad_id','bid', 'exp_num','adjusted_exp_num']].head(50))
    #
    # local_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    # path = '../../submission/' + local_time
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # submissionfilename = path + "/submission.csv"
    # test_sampleDF.to_csv(submissionfilename, index=False, encoding="utf-8", header=None,
    #                      columns=["sample_id", "adjusted_exp_num"])

    print('mean:  ', np.mean(test_sampleDF['exp_num']))
    print('sum:  ', np.sum(test_sampleDF['exp_num']))
    print('min:  ', np.min(test_sampleDF['exp_num']))
    print('max:  ', np.max(test_sampleDF['exp_num']))
    #
    # print('adjusted_exp_num')
    # print('mean:  ', np.mean(test_sampleDF['adjusted_exp_num']))
    # print('sum:  ', np.sum(test_sampleDF['adjusted_exp_num']))
    # print('min:  ', np.min(test_sampleDF['adjusted_exp_num']))
    # print('max:  ', np.max(test_sampleDF['adjusted_exp_num']))
    #
    # print(test_sampleDF.shape)
    return test_sampleDF

def zuiqiangguize_location_id_ad_id_winning_probability():
    # path = '../../data/total_data/process_train_data/train_data.csv'

    data = pd.read_csv(path ,  delimiter= '\t')
    # print(data.shape)
    # print(data.head())
    # print(data.isnull().sum())


    group_df = data.groupby(['ad_id','location_id'])\
        ['exp_num', 'ad_id_request_num'].\
        agg({'exp_num':'sum',
             'ad_id_request_num':'sum'}).reset_index()
    group_df['ad_id_winning_probability'] = group_df['exp_num'] / group_df['ad_id_request_num']
    del group_df['ad_id_request_num']
    del group_df['exp_num']
    gc.collect()
    # print('data聚合')
    # print(group_df.head())
    # print(group_df.shape)
    # test_sample_bid_path = '../../data/total_data/test_data/guize/simple_whole_test.txt'

    # test_sample_bid_path = '../../data/try_location_id/test_data/simple_whole_test.txt'
    test_sampleDF = pd.read_csv(simple_whole_test , usecols=['sample_id',
                                                    'ad_id', 'location_id','bid','ad_id_request_num','num_of_opponents'])
    # print('test_sampleDF')
    test_sampleDF.sort_values(by=['sample_id','ad_id', 'bid'], inplace=True)
    # print(test_sampleDF['sample_id'].value_counts())
    # print(test_sampleDF.head(20))
    # print(test_sampleDF.shape)

    group_df['ad_id'] = group_df['ad_id'].astype(int ,inplace = True )
    test_sampleDF['ad_id'] = test_sampleDF['ad_id'].astype(int, inplace=True)
    group_df['location_id'] = group_df['location_id'].astype(int, inplace=True)
    test_sampleDF['location_id'] = test_sampleDF['location_id'].astype(int, inplace=True)


    test_sampleDF = pd.merge(test_sampleDF, group_df, on = ['ad_id','location_id'], how= 'left')

    test_sampleDF.fillna(0, inplace=True)
    test_sampleDF['exp_num'] = test_sampleDF['ad_id_winning_probability'] * test_sampleDF['ad_id_request_num']
    # print(test_sampleDF.isnull().sum())
    test_sampleDF.sort_values(by=['ad_id','bid'], inplace= True )
    # print(test_sampleDF.head())
    # print(test_sampleDF.shape)

    test_sampleDF = test_sampleDF.groupby(['sample_id','ad_id', 'bid']) \
        ['exp_num']. \
        agg({'exp_num': 'sum'}).reset_index()

    # #
    print('mean:  ', np.mean(test_sampleDF['exp_num']))
    print('sum:  ', np.sum(test_sampleDF['exp_num']))
    print('min:  ', np.min(test_sampleDF['exp_num']))
    print('max:  ', np.max(test_sampleDF['exp_num']))
    #
    # print('adjusted_exp_num')
    # print('mean:  ', np.mean(test_sampleDF['adjusted_exp_num']))
    # print('sum:  ', np.sum(test_sampleDF['adjusted_exp_num']))
    # print('min:  ', np.min(test_sampleDF['adjusted_exp_num']))
    # print('max:  ', np.max(test_sampleDF['adjusted_exp_num']))
    #
    return test_sampleDF

# zuiqiangguize_location_id_ad_id_winning_probability
# mean 13.9207962314
# max 2010.0011
# min 0.0001
# sum 876564.6971

# zuiqiangguize_location_id_per_times_winning_rate
# mean:   10.3161716602
# sum:   649588.6971
# min:   0.0001
# max:   1446.0011

def zuiqiangguize_merge():
    print('*****************begin zuiqiangguize_merge begin*********************')
    higher_score_df = zuiqiangguize_location_id_ad_id_winning_probability()
    lower_score_df = zuiqiangguize_location_id_per_times_winning_rate()

    higher_score_df.rename(columns={'exp_num': 'higher_score_exp_num'}, inplace=True)
    lower_score_df.rename(columns={'exp_num': 'lower_score_exp_num'}, inplace=True)

    merge_df = pd.merge(higher_score_df, lower_score_df[ ['sample_id','lower_score_exp_num'] ], on=['sample_id'], how='left')
    merge_df['exp_num'] = (merge_df['higher_score_exp_num']) ** 0.6 * \
                          (merge_df['lower_score_exp_num']) ** 0.4
    merge_df = adjust_new_mom_exp_num(merge_df)

    print('adjusted_exp_num')
    print('mean:  ', np.mean(merge_df['adjusted_exp_num']))
    print('sum:  ', np.sum(merge_df['adjusted_exp_num']))
    print('min:  ', np.min(merge_df['adjusted_exp_num']))
    print('max:  ', np.max(merge_df['adjusted_exp_num']))

    merge_df = merge_df[["sample_id", "adjusted_exp_num"]]
    print('*****************end zuiqiangguize_merge end*********************')
    return merge_df

if __name__ == '__main__':
    # guize_C()
    # zuiqiangguize()
    # zuiqiangguize_location_id_ad_id_winning_probability()
    # zuiqiangguize_location_id_per_times_winning_rate()
    # zuiqiangguize_with_04_23()
    zuiqiangguize_merge()

    # path =  '../../data/try_location_id/train_data.csv'
    # data = pd.read_csv(path ,  delimiter= '\t')
    # print(data.shape)
    # print(data.head())
    # print(data.isnull().sum())
