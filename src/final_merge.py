### 测试集选取最后一天3.19作为验证集

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
from dataConfig import dataConfig
from read_data import *
from utils import *
from final_guize import *
config = dataConfig()
# file_path = config.file_path
logging.basicConfig(level=logging.INFO)

model_path = modelPath()
samplefilename = model_path.test_sample_bid_path
test_sample_bid_path = model_path.test_sample_bid_path

exposure_group_all_path = model_path.ad_id_train_path
saved_guize_data = model_path.test_sample_bid_path

# model_path = '../submission/2019-06-14 11:06/submission.csv'
# rule_path = '../submission/2019-06-12 10:13/submission.csv'

submissionfilename =  "../submission.csv"

high_score_path = '../submission/lgb_without_dandiaoxing/submission.csv'
lower_score_path = '../submission/xgb_without_dandiaoxing/submission.csv'

def detect_submission(submissionfilename):
    test_sampledf_result = pd.read_csv(submissionfilename,
                                       sep=",", header=None, names=["ad_id", "exp_num"])
    print(test_sampledf_result.shape)
    print(test_sampledf_result.isnull().sum())
    # print(test_sampledf_result.head())
    print( 'mean' , np.mean(test_sampledf_result['exp_num'].values) )
    print( 'max' , np.max(test_sampledf_result['exp_num'].values) )
    print( 'min' , np.min(test_sampledf_result['exp_num'].values) )
    print( 'sum' , np.sum(test_sampledf_result['exp_num'].values) )
    # data = load_clean_data()
    # # print(data.columns)
    # train = data[data['flag']!=3]
    # print( np.mean(train['exp_num'].values) )
    # print(test_sampledf_result.shape)
    # get_testA_new_ad_id_and_old_ad_id_shape()

def detect_submission_with_out_dandiaoxing(submissionfilename):
    # # sample_id,ad_id,target_conversion_type,charge_type,bid,exp_num
    # xgb_df = pd.read_csv(lower_score_path, usecols = ['sample_id','exp_num'])
    # xgb_df.to_csv('/home/hadoop/jwt/PycharmProject/Tencent_second/submission/xgb_2019-06-10 17:31_without_dandiaoxing/submission.csv', index = None, header = None)

    test_sampledf_result = pd.read_csv(submissionfilename, usecols=["ad_id", "exp_num"])
    print(test_sampledf_result.shape)
    # print(test_sampledf_result.head())
    print( 'mean' , np.mean(test_sampledf_result['exp_num'].values) )
    print( 'max' , np.max(test_sampledf_result['exp_num'].values) )
    print( 'min' , np.min(test_sampledf_result['exp_num'].values) )
    print( 'sum' , np.sum(test_sampledf_result['exp_num'].values) )
    # data = load_clean_data()
    # # print(data.columns)
    # train = data[data['flag']!=3]
    # print( np.mean(train['exp_num'].values) )
    # print(test_sampledf_result.shape)
    # get_testA_new_ad_id_and_old_ad_id_shape()

def adjust_new_mom_exp_num(test):

    test['exp_num'] = test['exp_num'].map(lambda x : max(x, 0))
    test['sample_id'] = test['sample_id'].astype(int)
    test_sampleDF = test

    test.sort_values(by=["ad_id", "bid"], inplace=True)
    temp_test_sampleDF = test.copy()
    bid = temp_test_sampleDF.set_index('sample_id')[['ad_id', 'bid']].groupby('ad_id')['bid'].apply(
        lambda row: pd.Series(dict(zip(row.index, row.rank() * 0.0001)))).reset_index()

    test_sampleDF['exp_num'] = test_sampleDF['exp_num'].apply(lambda x: round(x, 0))
    test_sampleDF['adjusted_exp_num'] = test_sampleDF['exp_num'].values + bid['bid'].values
    return test_sampleDF

def read_test_old_ad_id_data(test_sample_path = '../data/test_old.csv'):
    test_sampleDF = pd.read_csv(test_sample_path)
    return test_sampleDF

def read_test_new_ad_id_data(test_sample_path = '../data/test_new.csv'):
    test_sampleDF = pd.read_csv(test_sample_path)
    return test_sampleDF

def split_test_data():
    # exposure_group_all_path = ('../data/total_data/process_train_data/train_data.csv')

    train  = pd.read_csv(exposure_group_all_path , delimiter='\t')
    # print(train.head())

    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']

    test = pd.read_csv(saved_guize_data, delimiter='\t', \
                                header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int})

    test['request_day'] = '2019-04-23'

    history_ad_id = pd.DataFrame({ 'ad_id' : list(set(train.ad_id) & set(test.ad_id)) } )
    new_ad_id = pd.DataFrame(  {'ad_id': list(set(test.ad_id) - set(train.ad_id)) } )
    # print(history_ad_id.head())

    history_ad_id['new_or_old'] = 0
    new_ad_id['new_or_old'] = 1
    history_new_ad_df = pd.concat([history_ad_id, new_ad_id], axis=0, ignore_index=True)
    test = pd.merge(test, history_new_ad_df, how='left', on='ad_id')
    test_old = test[test.new_or_old == 0]
    test_new = test[test.new_or_old == 1]
    # test_old.to_csv("test_old.csv", index=False)
    # test_new.to_csv("test_new.csv", index=False)
    print(train.shape)
    print(test.shape)
    print(test_old.shape)
    print(test_new.shape)
    return test_old, test_new


def get_new_ad_id_and_old_ad_id_shape():
    test_old_ad_id_df = read_test_old_ad_id_data()
    test_new_ad_id_df = read_test_new_ad_id_data()
    print('old shape', test_old_ad_id_df.shape)
    print( len(set(test_old_ad_id_df['ad_id'].values)) )

    print('new shape', test_new_ad_id_df.shape)
    print( len(set(test_new_ad_id_df['ad_id'].values)) )
    test = read_test_data()
    print('test shape', test.shape)
    test.sort_values(by=['ad_id','ad_bid'],inplace=True)
    # print(test[['ad_id','ad_bid']].head(50))

# def get_testA_new_ad_id_and_old_ad_id_shape():
#     test_old_ad_id_df = read_test_old_ad_id_data( test_sample_path = '../data/test_old.csv')
#     test_new_ad_id_df = read_test_new_ad_id_data(test_sample_path = '../data/test_new.csv')
#     print('old shape', test_old_ad_id_df.shape)
#     print( 'old ad_id num' , len(set(test_old_ad_id_df['ad_id'].values)) )
#
#     print('new shape', test_new_ad_id_df.shape)
#     print( 'new ad_id num ',len(set(test_new_ad_id_df['ad_id'].values)) )
#
#
#     test_track_log_path = '../data/total_data/BTest/BTest_tracklog_20190424.txt'
#     final_select_test_request_path = '../data/total_data/BTest/Btest_select_request_20190424.out'
#     test_sample_bid_path = '../data/total_data/BTest/Btest_sample_bid.out'
#
#     test_track_log_df, final_select_test_request_df, test_sample_bid_path_df = read_test_data(
#         test_track_log_path,
#         final_select_test_request_path,
#         test_sample_bid_path)
#
#     print('test_track_log_df   ', test_track_log_df.shape)
#     print('final_select_test_request_df    ', final_select_test_request_df.shape)
#     print('test shape', test_sample_bid_path_df.shape)

def try8merge_getMonoScore(samplefilename, submissionfilename):
    # test_track_log_df, final_select_test_request_df, test_sample_bid_path_df = read_test_data(test_track_log_path,final_select_test_request_path
    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_samplefile = pd.read_csv(samplefilename, delimiter='\t', \
                                          header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int},
                                          usecols=['sample_id', 'ad_id', 'bid'])

    test_sampledf = pd.DataFrame(test_samplefile)

    # print(test_sampledf.shape)

    test_sampledf_result = pd.read_csv(submissionfilename,
                                       sep=",", header=None, names=["sample_id", "exp_num"])
    test_sampledf_resultdf = pd.DataFrame(test_sampledf_result)

    print("samplefilename:   ",test_sampledf.shape)
    print("submissionfilename:   ",test_sampledf_resultdf.shape)
    print('submission_data:   ')
    print(test_sampledf_result.head(10))
    test_sampledf = pd.merge(test_sampledf, test_sampledf_resultdf, how='left', left_on='sample_id', right_on='sample_id')
    test_sampledf.sort_values(by=["ad_id", "bid"], inplace=True)

    # 作为基准
    standard = test_sampledf.groupby(by='ad_id').head(1)
    standard.rename(columns={'sample_id': 'base_sample_id', 'bid': 'base_bid', 'exp_num': 'base_exp_num'}, inplace=True)

    test_sampledf = pd.merge(test_sampledf, standard, how="left", left_on='ad_id', right_on='ad_id')
    test_sampledf['score'] = test_sampledf.apply(
        lambda x: (
                ((x['base_exp_num'] - x['exp_num']) * (x['base_bid'] - x['bid'])) /
                abs((x['base_exp_num'] - x['exp_num']) * (x['base_bid'] - x['bid']))
        )
        , axis=1
    )

    monoscore = test_sampledf.groupby(by='ad_id')['score'].mean().mean()
    print("score：" + str(60 * (monoscore + 1) / 2))

def my_get_save_submission(history_new_ad_df):
    # local_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    # path = '../submission/' + local_time
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # submissionfilename = "../submission.csv"
    history_new_ad_df.to_csv(submissionfilename, index=False, encoding="utf-8", header=None,
                                  columns=["sample_id", "adjusted_exp_num"])
    try8merge_getMonoScore(samplefilename, submissionfilename)


def model_merge():
    # ad_id location_id
    # high_score_path = '/home/hadoop/jwt/PycharmProject/Tencent_second/submission/2019-06-11 08:38_without_dandiaoxing/submission.csv'
    # lower_score_path = '/home/hadoop/jwt/PycharmProject/Tencent_second/submission/2019-06-14 10:53_without_dandiaoxing/submission.csv'
    print('**************begin model_merge begin*************************')
    higher_score_df = pd.read_csv(high_score_path)
    lower_score_df = pd.read_csv(lower_score_path)

    higher_score_df.rename(columns={'exp_num': 'higher_score_exp_num'}, inplace=True)
    lower_score_df.rename(columns={'exp_num': 'lower_score_exp_num'}, inplace=True)

    merge_df = pd.merge(higher_score_df, lower_score_df[ ['sample_id','lower_score_exp_num'] ], on=['sample_id'], how='left')
    merge_df['exp_num'] = (merge_df['higher_score_exp_num']) * 0.6 + \
                          (merge_df['lower_score_exp_num']) * 0.4
    merge_df = adjust_new_mom_exp_num(merge_df)
    # local_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    # path = '../submission/' + local_time
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # submissionfilename = path + "/submission.csv"
    # merge_df.to_csv(submissionfilename, index=False, encoding="utf-8", header=None,
    #                 columns=["sample_id", "adjusted_exp_num"])
    # getMonoScore(samplefilename, submissionfilename)
    # detect_submission(submissionfilename=submissionfilename)

    merge_df = merge_df[["sample_id", "adjusted_exp_num"]]
    print('**************end model_merge end*************************')
    return merge_df

def merge_test_old_and_new_data(model_merge_df, guize_merge_df):
    print('****************begin merge_test_old_and_new_data begin******************************')
    # test_sample_bid_path = '../data/total_data/test_sample_bid.out'
    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_sample_df = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                 header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int})


    # read_test_ad_id_data = pd.read_csv(model_path,
    #                                    names=['sample_id', 'adjusted_exp_num'])
    read_test_ad_id_data = model_merge_df
    print('model:   ')
    # read_test_new_ad_id_df = read_test_new_ad_id_data()
    test_old_ad_id_df, read_test_new_ad_id_df = split_test_data()

    read_test_ad_id_data = pd.merge(read_test_ad_id_data, test_sample_df, on='sample_id', how='left')

    new_ad_id = list(set(read_test_new_ad_id_df['ad_id'].values))
    read_test_new_ad_id_df = read_test_ad_id_data[read_test_ad_id_data['ad_id'].isin(new_ad_id)]


    # rule
    # test_old_ad_id_df = read_test_old_ad_id_data()
    # test_liusen_df = pd.read_csv(rule_path,
    #                              names=['sample_id', 'adjusted_exp_num'])
    test_liusen_df = guize_merge_df

    print('guize:   ')
    print(np.mean(test_liusen_df['adjusted_exp_num']))
    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_sample_df = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                 header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int})


    test_liusen_df = pd.merge(test_liusen_df, test_sample_df, on='sample_id', how='left')

    old_ad_id = list(set(test_old_ad_id_df['ad_id'].values))
    test_liusen_old_df = test_liusen_df[test_liusen_df['ad_id'].isin(old_ad_id)]

    print('old mean:  ', np.mean(test_liusen_old_df['adjusted_exp_num']))
    print('old shape:  ', test_liusen_old_df.shape)
    print('new mean:  ', np.mean(read_test_new_ad_id_df['adjusted_exp_num']))
    print('new shape:  ',read_test_new_ad_id_df.shape)
    history_new_ad_df = pd.concat([test_liusen_old_df, read_test_new_ad_id_df], axis=0, ignore_index=True)
    print('after mean:  ', np.mean(history_new_ad_df['adjusted_exp_num']))
    print('after shape: ',history_new_ad_df.shape)

    # 存储
    my_get_save_submission(history_new_ad_df)
    print('****************end merge_test_old_and_new_data end******************************')


def require_MonoScore(submissionfilename):
    print('*********begin require_MonoScore begin *********')
    getMonoScore(samplefilename, submissionfilename)
    print('*********end require_MonoScore end *********')

if __name__ == '__main__':
    # lower_score_path = '/home/hadoop/jwt/PycharmProject/Tencent_second/submission/2019-06-13 01:14_xgb/submission.csv'
    # detect_submission(    submissionfilename = lower_score_path )
    guize_merge_df = zuiqiangguize_merge()
    model_merge_df = model_merge()
    merge_test_old_and_new_data(model_merge_df , guize_merge_df)
    # detect_submission(submissionfilename=submissionfilename)
    # lgb_model_path = '../submission/lgb_without_dandiaoxing/submission.csv'
    # xgb_model_path = '../submission/xgb_without_dandiaoxing/submission.csv'
    # detect_submission_with_out_dandiaoxing(lgb_model_path)
    # detect_submission_with_out_dandiaoxing(xgb_model_path)
