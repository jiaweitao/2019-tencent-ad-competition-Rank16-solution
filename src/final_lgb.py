

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
from read_data import *
from load_clean_data import *
from dataConfig import dataConfig
logging.basicConfig(level=logging.INFO)

model_path = modelPath()
saved_get_recent_ecpm_path = model_path.saved_get_recent_ecpm_path
get_recent_ecpm_path  = model_path.get_recent_ecpm_path
test_sample_bid_path = model_path.test_sample_bid_path
Log_file  = model_path.Log_file
samplefilename  = model_path.samplefilename

without_dandiaoxing_path = '../submission/' + 'lgb_without_dandiaoxing'
path = '../submission/' + 'lgb'

###########################################
#                    #

###########################################

###########################################
#                             #
###########################################

def generate_one_hot_features(data, categorical_columns, numerical_columns):
    logging.info("***begin*******generate_one_hot_features****begin*******")
    # categorical_columns = [f for f in train_columns if f not in ['ad_size']]
    print("num：  ")
    for col in numerical_columns:
        print(col)
    print("one_hot:  ", categorical_columns)
    # label encoder
    for f in categorical_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))

    train = data[data['flag'] != 3]
    test = data[data['flag'] == 3]

    train_num = data[data['flag'] != 3].shape[0]
    test_num = data[data['flag'] == 3].shape[0]

    train_y = data.loc[data['flag'] != 3, 'exp_num'].map(log_convert)

    return_columns = numerical_columns
    # try:
    X = train[numerical_columns].values
    # except:
    #    print(numerical_columns)
    #    print(train.columns)
    x_test = test[numerical_columns].values
    enc = OneHotEncoder()
    for f in categorical_columns:
        enc.fit(data[f].values.reshape(-1, 1))
        # print(X.dtypes, ' ', train[f].dtypes )
        temp_X = enc.transform(train[f].values.reshape(-1, 1))
        X = sparse.hstack((X, temp_X), 'csr')
        temp_x_test = enc.transform(test[f].values.reshape(-1, 1))
        # temp_x_test = enc.transform(test[f].values.reshape(-1, 1)).astype(object)
        x_test = sparse.hstack((x_test, temp_x_test), 'csr')
        one_hot_col = [f + '_' + str(i) for i in range(0, enc.transform(train[f].values.reshape(-1, 1)).shape[1])]
        return_columns.extend(one_hot_col)
    y = train_y.values

    logging.info("***end*******generate_one_hot_features****end*******")
    return X, y, x_test, return_columns


def generate_data_log_competitive_and_user_cover_feature(data ):

    print("********** begin generate_data_log_competitive_and_user_cover_feature begin******************")

    data_log = pd.read_csv(Log_file, delimiter='\t')

    print(data_log.isnull().sum())
    return__competitive_and_people_cover_columns = []

    # data, competitve2_locaiont_columns = generate_competitve2_locaiont_id(data_log,data)
    # generate_cover2_user_id_count
    data , adsize_locaiont_columns = generate_cover1_adsize_locaiont_id(data_log,data)

    return__competitive_and_people_cover_columns.extend(adsize_locaiont_columns)

    print("********** end generate_data_log_competitive_and_user_cover_feature end******************")
    t = (time.time() - start) / 60
    print()
    print("generate_data_log_competitive_and_user_cover_feature running time: {:.2f} minutes\n".format(t))
    return data, return__competitive_and_people_cover_columns

def generate_cover1_adsize_locaiont_id(data_log, data):
    print("********** begin generate_cover1_adsize_locaiont_id begin******************")
    start = time.time()
    cross_categorical_columns_cover1_ad_size = ['ad_size']
    competitive_power_cover1_ad_size = ['location_id']
    for col in cross_categorical_columns_cover1_ad_size:
        data_log[ col ] = data_log[col].astype(int , inplace = True )
        data[ col ] = data[col].astype(int, inplace=True)
    col_select = [
        'ad_size_location_id_kfold_unique',
    ]
    data['index'] = list(range(data.shape[0]))

    for col in cross_categorical_columns_cover1_ad_size:
        for target_encoding in competitive_power_cover1_ad_size:
            col_name = col + '_' + target_encoding + '_kfold_unique'
            if col_name in col_select:
                print('11111111111111')
                # data = statis_count_of_feat(data_log, data, col, target_encoding, col_name)
                df_cv = data[['index', col, 'flag']].copy()

                test = df_cv.loc[df_cv['flag'] == 3, :]
                # print('test1:   ',test.isnull().sum() )
                test = statis_count_of_feat(data_log, test, col, target_encoding, col_name)
                # print('test2:   ', test.isnull().sum())
                train = df_cv.loc[df_cv['flag'] != 3, :].reset_index(drop=True)
                folds = KFold(n_splits=5, random_state=2018, shuffle=True)
                train['fold'] = None
                for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train)):
                    train.loc[val_idx, 'fold'] = fold_

                df_stas_feat = None
                for fold_, (trn_idx, val_idx) in enumerate(folds.split(data_log, data_log)):
                    Log_trn = data_log.iloc[trn_idx]
                    X_val = train[train['fold'] == fold_]
                    print('Log_trn: ',Log_trn.shape)
                    print('Log_trn: ',Log_trn.isnull().sum())
                    X_val = statis_count_of_feat(Log_trn, X_val, col, target_encoding, col_name)
                    print('X_val: ',X_val.shape)
                    print('X_val:  ',X_val.isnull().sum())
                    df_stas_feat = pd.concat([df_stas_feat, X_val], axis=0)
                df_stas_feat = pd.concat([df_stas_feat, test], axis=0)
                df_stas_feat.drop([col, 'flag'], axis=1, inplace=True)
                data = pd.merge(data, df_stas_feat, how='left', on='index')
    # print(data[col_select[0]].isnull().sum())
    del data_log
    gc.collect()

    print("********** end generate_cover1_adsize_locaiont_id end******************")
    t = (time.time() - start) / 60
    print()
    print("generate_cover1_adsize_locaiont_id running time: {:.2f} minutes\n".format(t))

    return data, col_select

def get_excepted_exp_num_columns(data):
    print("********** begin get_excepted_exp_num_columns begin******************")

    feats = ['ad_id']
    # competitive_power = [ 'ad_id_winning_probability' ]
    col_select = []
    for day_num in range(1, 4):
        for col in feats:
                target_col_name = 'ad_id_winning_probability'
                data, temp_columns = create_rencent_competitve_power_lagging(data, data, day_num, target_col_name,
                                                                             col)
                col_select.append(temp_columns)
    result_col = []
    for col in col_select:
        target_col_name = col + '_excepted_exp_num'
        data[target_col_name] = data[col] *  data['ad_id_request_num']
        result_col.append(target_col_name)

    result_col = result_col + col_select

    # data['ad_id_winning_probability_lag_max'] = data.apply(get_new_Max, axis=1, args=('lagging1', 'lagging2', 'lagging3',))
    # data['ad_id_winning_probability_lag_min'] = data.apply(get_new_Min, axis=1, args=('lagging1', 'lagging2', 'lagging3',))
    # data['ad_id_winning_probability_lag_mean'] = data.apply(get_new_Mean, axis=1, args=('lagging1', 'lagging2', 'lagging3',))

    data['ad_id_winning_probability_lag_max'] = data.apply(get_new_Max, axis=1,
                                                           args=(col_select[0], col_select[1], col_select[2],))
    data['ad_id_winning_probability_lag_min'] = data.apply(get_new_Min, axis=1,
                                                           args=(col_select[0], col_select[1], col_select[2],))
    data['ad_id_winning_probability_lag_mean'] = data.apply(get_new_Mean, axis=1,
                                                            args=(col_select[0], col_select[1], col_select[2],))
    col_select = ['ad_id_winning_probability_lag_mean']

    for col in col_select:
        target_col_name = col + '_excepted_exp_num'
        data[target_col_name] = data[col] *  data['ad_id_request_num']
        result_col.append(target_col_name)
    result_col = result_col + col_select

    print("********** end get_excepted_exp_num_columns end******************")

    return data,result_col

def get_recent_fail_num(data):
    print("********** begin get_fail_num begin******************")

    data['request_fail_num'] = data['ad_id_request_num'].values - data['exp_num'].values
    feats = ['ad_id','ad_acc_id', 'ad_size', 'ad_trade_id', 'good_class', 'good_id']
    # competitive_power = [ 'ad_id_winning_probability' ]
    col_select = []
    for day_num in range(1, 4):
        for col in feats:
                target_col_name = 'request_fail_num'
                data, temp_columns = create_rencent_competitve_power_lagging(data, data, day_num, target_col_name,
                                                                             col)
                col_select.append(temp_columns)


    print("********** end get_fail_num end******************")
    return data, col_select

def get_kflod_statstic_data(data):
    print("********** begin get_kflod_statstic_data begin******************")
    start = time.time()

    cross_categorical_columns = ['ad_id','ad_trade_id', 'ad_acc_id','ad_size',
                                 'good_id', 'good_class']

    competitive_power = ['exp_num','request_fail_num', 'ad_id_winning_probability' ]

    col_select = []
    data['index'] = list(range(data.shape[0]))

    for col in cross_categorical_columns:
        for target_encoding in competitive_power:
            col_name = col + '_' + target_encoding + '_kfold_mean'
            col_select.append(col_name)

            df_cv = data[['index', col, target_encoding, 'flag']].copy()
            train = df_cv.loc[df_cv['flag'] != 3, :].reset_index(drop=True)
            test = df_cv.loc[df_cv['flag'] == 3, :]
            temp = train.groupby(col, as_index=False)[target_encoding].agg(
                {col_name: 'mean'})

            test = pd.merge(test, temp, on=col, how='left')
            # test[col_name].fillna(test[col_name].mean(), inplace=True)

            df_stas_feat = None
            kf = KFold(n_splits=5, random_state=2018, shuffle=True)
            for train_index, val_index in kf.split(train):
                X_train = train.loc[train_index, :]
                X_val = train.loc[val_index, :]

                X_val = statis_feat(X_train, X_val, col, target_encoding)
                df_stas_feat = pd.concat([df_stas_feat, X_val], axis=0)

            df_stas_feat = pd.concat([df_stas_feat, test], axis=0)
            df_stas_feat.drop([col, target_encoding, 'flag'], axis=1, inplace=True)
            data = pd.merge(data, df_stas_feat, how='left', on='index')

    gc.collect()

    print("********** end get_kflod_statstic_data end******************")
    t = (time.time() - start) / 60
    print()
    print("get_kflod_statstic_data running time: {:.2f} minutes\n".format(t))

    return data, col_select

def get_rencent_winning_probability_exp_num(data):
    print("********** begin get_rencent_winning_probability_exp_num begin******************")
    start = time.time()
    # feats = ['ad_acc_id','good_id']
    feats = ['ad_acc_id', 'ad_size', 'ad_trade_id','good_class', 'good_id']
    # , 'ad_id_winning_probability'
    competitive_power = [ 'exp_num' , 'ad_id_winning_probability']

    col_select = []
    for day_num in range(1,4):
        for col in feats:
            for target_col_name in competitive_power:
                data, temp_columns = create_rencent_competitve_power_lagging(data, data, day_num, target_col_name,
                                                                       col)
                col_select.append(temp_columns)

    for i in range(0, len(feats)):
            for j in range(i+1, len(feats)):
                # feat_name = feats[i] + "_" + feats[i + j + 1]
                cross_categorical_columns = [feats[i], feats[j]]
                # print(feat_name)
                for day_num in range(1,4):
                    for target_col_name in competitive_power:
                        col_name ='_'.join(cross_categorical_columns) + '_lagging_'  + target_col_name + '_' + str(i)
                        # if col_name in col_select:
                        data, temp_columns = \
                           create_col_list_rencent_competitive_lagging(data,
                                                                    data, day_num, target_col_name,
                                                                    cross_categorical_columns)

                        col_select.append(temp_columns)
    # for i in range(0, len(feats)):
    #     for j in range(i + 1, len(feats)):
    #         for k in range(j + 1, len(feats)):
    #             # feat_name = feats[i] + "_" + feats[i + j + 1]
    #             for day_num in range(1, 4):
    #                 cross_categorical_columns = [feats[i], feats[j], feats[k]]
    #                 # print(feat_name)
    #                 for target_col_name in competitive_power:
    #                     col_name = '_'.join(cross_categorical_columns) + '_lagging_' + target_col_name + '_' + str(i)
    #                     # if col_name in col_select:
    #                     data, temp_columns = \
    #                         create_col_list_rencent_competitive_lagging(data,
    #                                                                     data, day_num, target_col_name,
    #                                                                     cross_categorical_columns)
    #                     col_select.append(temp_columns)

    # for col in col_select:
    #     data[col].fillna(0, inplace=True)
    #     print(col)
    print(data.columns)
    # print(data[col_select].head())
    print("********** end get_rencent_winning_probability_exp_num end******************")
    t = (time.time() - start) / 60
    print()
    print("get_rencent_competitve_power_columns running time: {:.2f} minutes\n".format(t))

    return data, col_select

def get_recent_ecpm(saved_file, columns_path ):

    if os.path.exists(saved_file):
        data = pd.read_csv(saved_file)
        columns = pickle.load(open(columns_path, 'rb'))
        return data, columns
    else:
        start = time.time()
        data = my_load_clean_with_dynamic_data()
        data, lagging_columns = create_new_lagging(data)

        print("********** begin get_recent_ecpm_columns begin******************")
        feats = ['ad_id', 'ad_acc_id', 'ad_size', 'ad_trade_id', 'good_class', 'good_id']
        competitive_power = ['bidding_price_mean','pctr_mean','quality_ecpm_mean','totalEcpm_mean']
        col_select = []
        # i = 1
        for col in feats:
            for target_col_name in competitive_power:
                for day_num in range(1,4):
                    data, temp_columns = create_rencent_competitve_power_lagging(data, data, day_num, target_col_name,
                                                                                 col)
                    col_select.append(temp_columns)

        data['ad_id_totalEcpm_mean_lag_max'] = data.apply(get_new_Max, axis=1,
                                                          args=('ad_id_lagging__totalEcpm_mean_1',
                                                                'ad_id_lagging__totalEcpm_mean_2',
                                                                'ad_id_lagging__totalEcpm_mean_3',))
        data['ad_id_totalEcpm_mean_lag_min'] = data.apply(get_new_Min, axis=1,
                                                          args=('ad_id_lagging__totalEcpm_mean_1',
                                                                'ad_id_lagging__totalEcpm_mean_2',
                                                                'ad_id_lagging__totalEcpm_mean_3',))
        data['ad_id_totalEcpm_mean_lag_mean'] = data.apply(get_new_Mean, axis=1,
                                                           args=('ad_id_lagging__totalEcpm_mean_1',
                                                                 'ad_id_lagging__totalEcpm_mean_2',
                                                                 'ad_id_lagging__totalEcpm_mean_3',))
        add_columns = ['ad_id_totalEcpm_mean_lag_max', 'ad_id_totalEcpm_mean_lag_min',  'ad_id_totalEcpm_mean_lag_mean']
        col_select = col_select + add_columns

        feats = ['ad_acc_id', 'ad_size', 'ad_trade_id', 'good_class', 'good_id']
        competitive_power = ['bidding_price_mean', 'pctr_mean', 'quality_ecpm_mean', 'totalEcpm_mean']

        for i in range(0, len(feats)):
            for j in range(i + 1, len(feats)):
                cross_categorical_columns = [feats[i], feats[j]]
                # print(feat_name)
                for k in range(1, 4):
                    for target_col_name in competitive_power:
                        # col_name = '_'.join(cross_categorical_columns) + '_lagging_' + target_col_name + '_' + str(i)
                        # if col_name in col_select:
                        data, temp_columns = create_col_list_rencent_competitive_lagging(data,
                                                                        data, k, target_col_name,
                                                                        cross_categorical_columns)

                        col_select.append(temp_columns)
        data.to_csv(saved_file)
        pickle.dump((col_select), open(columns_path, 'wb'))

        print("********** end get_recent_ecpm_columns end******************")
        t = (time.time() - start) / 60
        print()
        print("get_recent_ecpm_columns running time: {:.2f} minutes\n".format(t))

        return data,  lagging_columns + col_select

def get_recent_ad_id_competitive_compare_columns(data):
    print("********** begin get_recent_ad_id_competitive_columns begin******************")
    start = time.time()
    feats = ['ad_acc_id', 'ad_id',  'ad_trade_id','good_id']
    competitive_power = ['ad_id_winning_probability']

    col_select = []
    i = 1
    for col in feats:
        for target_col_name in competitive_power:
            col_name = col + '_' + target_col_name + '_lagging_' + '_' + str(i)
            # if col_name in col_select:
            data, temp_columns = create_rencent_competitve_power_lagging(data, data, i, target_col_name,
                                                                         col)
            col_select.append(temp_columns)

    compete_columns = [ 'compete_ad_total_ecpm_merge_mean' ]

    return_columns = []
    for i in range( 0 , len(col_select)):
        col = col_select[i]
        # index = i * 3
        # for j in range(0,1):
        j = 0
        diff_col = compete_columns[  j ]
        col_name = col + '_diff_' + diff_col
        data[col_name] = data[col].values - data['compete_ad_total_ecpm_merge_max'].values
        return_columns.append(col_name)

    print("********** end get_rencent_competitve_power_columns end******************")
    t = (time.time() - start) / 60
    print()
    print("get_recent_ad_id_competitive_columns running time: {:.2f} minutes\n".format(t))

    return data, return_columns

def get_delata(data):
    print("********** begin get_delata begin******************")
    start = time.time()

    data['delta_request_day_create_time'] = pd.to_datetime(data['request_day']) - pd.to_datetime(data['create_time'])
    data['delta_request_day_create_time'] = data['delta_request_day_create_time'].map( lambda delta : delta.days)

    data['begin_time'] = '20190409'
    data['start_time'] = pd.to_datetime(data['request_day']) - pd.to_datetime(data['begin_time'])

    data['create_time'] = pd.to_datetime(data['create_time'])
    data['create_time'] = data['create_time'].dt.strftime('%Y%m%d')

    data['request_day'] = pd.to_datetime(data['request_day'])
    data['request_day'] = data['request_day'].dt.strftime('%Y%m%d')

    data['delta_request_day_create_time'] = data['delta_request_day_create_time'].astype(int)
    data['start_time'] = data['start_time'].astype(int)
    data['create_time'] = data['create_time'].astype(int)
    data['request_day'] = data['request_day'].astype(int)
    print("********** end get_delata end******************")
    t = (time.time() - start) / 60
    print()
    print("create_lagging running time: {:.2f} minutes\n".format(t))
    # start_time request_day
    return data, ['delta_request_day_create_time','create_time' , 'request_day']

def saved_features(data, numerical_columns , categorical_columns ):
    dataconfig = dataConfig()
    data = data[ numerical_columns + categorical_columns + ['flag'] + ['exp_num'] ]

    saved_file = dataconfig.lgb_saved_file
    return_numerical_columns_path = dataconfig.lgb_return_numerical_columns_path
    return_categorical_columns_path = dataconfig.lgb_return_categorical_columns_path
    if os.path.exists(saved_file):
        logging.info(saved_file + ' already exists')
        return
    data.to_csv(saved_file, index = None)
    pickle.dump((numerical_columns), open(return_numerical_columns_path, 'wb'))
    pickle.dump((categorical_columns), open(return_categorical_columns_path, 'wb'))

def load_feautes():
    dataconfig = dataConfig()
    saved_file = dataconfig.lgb_saved_file
    return_numerical_columns_path = dataconfig.lgb_return_numerical_columns_path
    return_categorical_columns_path = dataconfig.lgb_return_categorical_columns_path

    if not os.path.exists(saved_file):
        logging.info(saved_file + ' not exists')
        return
    data = pd.read_csv(saved_file)
    numerical_columns = pickle.load(open(return_numerical_columns_path, 'rb'))
    categorical_columns = pickle.load(open(return_categorical_columns_path, 'rb'))
    return data, numerical_columns, categorical_columns

def generate_features():
    # data = load_clean_190W_data()
    # data = my_load_clean_data()

    print("********** begin generate_features begin******************")
    start = time.time()
    # model_path = modelPath()
    # saved_file = model_path.saved_file
    dataconfig = dataConfig()
    saved_file = dataconfig.xgb_saved_file

    if os.path.exists(saved_file):
        data, numerical_columns, categorical_columns = load_feautes()
        print(set(data['request_day']))
        print('data:  ', data.shape)
    else:
        # new
        data, lagging_columns_get_recent_ecpm_columns = get_recent_ecpm(saved_get_recent_ecpm_path, get_recent_ecpm_path )

        data, request_faliure_num_columns = get_recent_fail_num(data)

        data, statis_ad_size_location_id_columns = generate_data_log_competitive_and_user_cover_feature(data )

        data, kflod_winning_probability_columns = get_kflod_statstic_data(data )
        print('*********kflod_winning_probability_columns************')
        print(kflod_winning_probability_columns)
        print('*********kflod_winning_probability_columns************')
        data, rencent_winning_probability_and_exp_num_columns = get_rencent_winning_probability_exp_num(
            data )
        data , excepted_exp_num_columns = get_excepted_exp_num_columns(data)

        # data, recent_ad_id_competitive_compare_columns = get_recent_ad_id_competitive_compare_columns(data)
        data, diff_reqest_create_columns = get_delata(data )

        numerical_columns =  lagging_columns_get_recent_ecpm_columns  +\
                             rencent_winning_probability_and_exp_num_columns + \
                             excepted_exp_num_columns +\
                             diff_reqest_create_columns + \
                             ['ad_id_request_num'] + \
                             statis_ad_size_location_id_columns + \
                             request_faliure_num_columns + \
                             kflod_winning_probability_columns

        categorical_columns = ['ad_acc_id','ad_id', 'ad_size', 'ad_trade_id',
                               'good_class', 'good_id'] +  ['charge_type']


        saved_features(data, numerical_columns , categorical_columns)

    train = data[ data['flag'] != 3 ]
    test  = data[ data['flag'] == 3 ]
    print('num:  ', numerical_columns)
    print('cat:   ',categorical_columns)
    # print( data[ categorical_columns + numerical_columns ].dtypes)
    print(data.shape)
    print('cat')
    print(data[categorical_columns ].isnull().sum())
    print('num')
    print(data[numerical_columns].isnull().sum() )
    # print('rencent_winning_probability_and_exp_num_columns')
    # print(data[rencent_winning_probability_and_exp_num_columns].isnull().sum() )
    # print(data[statis_ad_size_location_id_columns].isnull().sum() )

    print(train.shape)
    print('cat')
    print(train[categorical_columns ].isnull().sum())
    print('num')
    print(train[numerical_columns].isnull().sum() )
    # print('rencent_winning_probability_and_exp_num_columns')
    # print(train[rencent_winning_probability_and_exp_num_columns].isnull().sum() )
    # print(train[statis_ad_size_location_id_columns].isnull().sum() )
    # print( test[ categorical_columns + numerical_columns ].dtypes)
    print(test.shape)
    print('cat')
    print(test[categorical_columns ].isnull().sum())
    print('num')
    print(test[numerical_columns].isnull().sum() )
    # print('rencent_winning_probability_and_exp_num_columns')
    # print(test[rencent_winning_probability_and_exp_num_columns].isnull().sum() )
    # print(test[statis_ad_size_location_id_columns].isnull().sum() )

    # train = data[ data['flag'] != 3 ][0:1000]
    # test = data[data['flag']  == 3 ]
    # data = pd.concat( [train,test], axis =0 )

    X, y, x_test, return_columns = generate_one_hot_features(data, categorical_columns, numerical_columns)
    # X, y, x_test, return_columns = generate_cv_features(data, X, y, x_test, return_columns)

    print("********** end generate_features end******************")
    t = (time.time() - start) / 60
    print("generate_features running time: {:.2f} minutes\n".format(t))

    return X, y, x_test, return_columns


###########################################
#            模型部分                      #
###########################################

def lgb_kfold_predict(X, y, x_test, return_columns):
    logging.info("***begin*******lgb_kfold_one_hot_predict****begin*******")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 200,
        # 'num_leaves': 150,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_seed': 0,
        'bagging_freq': 1,
        'verbose': 1,
        'reg_alpha': 1,
        'reg_lambda': 2,
        'save_binary': True
    }

    training_time = 0
    smape_five = 0
    kf = KFold(n_splits=5, random_state=2333, shuffle=True)

    oof = np.zeros(X.shape[0])
    predictions = np.zeros(x_test.shape[0])

    saved_model = []
    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        t0 = time.time()
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        logging.info("this round shape:   ", X_train.shape, y_train.shape)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=50,
                        #         fobj=exp_loss,
                        )

        smape_five = smape_five + smape(y_val, gbm.predict(X_val, num_iteration=gbm.best_iteration))
        print("{} round offline validation smape: {:.4f}".format(i + 1,
                                                                 smape(y_val, gbm.predict(X_val,
                                                                                          num_iteration=gbm.best_iteration))))
        oof[val_index] = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        predictions += pd.Series(gbm.predict(x_test, num_iteration=gbm.best_iteration)).map(
            exp_convert).values / kf.n_splits

        t = (time.time() - t0) / 60
        training_time += t
        saved_model.append(gbm)
        print("This round cost time: {:.2f} minutes".format(t))

    print("smape 5 Fold: ", smape_five)
    print("CV score: {:<8.5f}".format(smape(oof, y)))
    print("Total training time cost: {:.2f} minutes".format(training_time))


    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int})

    # test = read_test_data()
    test['exp_num'] = predictions

    path = without_dandiaoxing_path

    if not os.path.exists(path):
        os.mkdir(path)
    submissionfilename = path + "/submission.csv"
    test.to_csv(submissionfilename, index=False)

    logging.info("***end*******lgb_kfold_one_hot_predict****end*******")

    return test ,gbm

def adjust_new_mom_exp_num(test, saved_model, return_columns):
    def process_fuzhi(x):
        if (x['exp_num'] < 0):
            return x['spare_exposure']
        else:
            return x['exp_num']

    # test['exp_num_process_fuzhi'] = test.apply(process_fuzhi, axis=1)

    test['exp_num'] = test['exp_num'].map(lambda x : max(x, 0))

    test['sample_id'] = test['sample_id'].astype(int)
    test_sampleDF = test
    print('test_sampleDF: ')
    print(test_sampleDF['exp_num'].head())

    test.sort_values(by=["ad_id", "bid"], inplace=True)
    temp_test_sampleDF = test.copy()
    bid = temp_test_sampleDF.set_index('sample_id')[['ad_id', 'bid']].groupby('ad_id')['bid'].apply(
        lambda row: pd.Series(dict(zip(row.index, row.rank() * 0.0001)))).reset_index()

    test_sampleDF['exp_num'] = test_sampleDF['exp_num'].apply(lambda x: round(x, 0))
    test_sampleDF['adjusted_exp_num'] = test_sampleDF['exp_num'].values + bid['bid'].values
    print('test_sampleDF: ')
    print(test_sampleDF['adjusted_exp_num'].head())

    with_adjust_exp_num_df = test_sampleDF

    if not os.path.exists(path):
        os.mkdir(path)
    saved_result(saved_model, with_adjust_exp_num_df, path, return_columns)

def require_MonoScore():
    # samplefilename = '../data/total_data/test_data/simple_whole_test.txt'
    submissionfilename = path + "/submission.csv"
    getMonoScore(samplefilename, submissionfilename)

if __name__ == '__main__':
    start = time.time()
    X, y, x_test, return_columns = generate_features()
    # X_train,up_train,X_val,up_val,X_whole,up_whole,x_test
    print("train shape:   ", X.shape, y.shape)

    test, saved_gbm = lgb_kfold_predict(X, y, x_test, return_columns)
    # test, saved_gbm = lgb_whole_predict(X, y, x_test, return_columns)

    print(test.head())
    print(test.dtypes)

    # t = (time.time() - start) / 60
    # adjust_new_mom_exp_num(test, saved_gbm, return_columns)
    # require_MonoScore()

    print("running time: {:.2f} minutes\n".format(t))

