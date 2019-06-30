from datetime import datetime ,timedelta
import pandas as pd
import numpy as np
import pickle, os, jieba, time, gc, re

import math
import matplotlib.pylab as plt
import lightgbm as lgb
import logging
logging.basicConfig(level=logging.INFO)

def getMonoScore(samplefilename, submissionfilename):
    # test_track_log_df, final_select_test_request_df, test_sample_bid_path_df = read_test_data(test_track_log_path,final_select_test_request_path
    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_samplefile = pd.read_csv(samplefilename, delimiter='\t', \
                                          header=None, names=names, dtype={"sample_id": int
            , 'ad_id': int, "target_conversion_type": int, 'charge_type': int, "bid": int},
                                          usecols=['sample_id', 'ad_id', 'bid'])

    test_sampledf = pd.DataFrame(test_samplefile)

    test_sampledf_result = pd.read_csv(submissionfilename,
                                       sep=",", header=None, names=["sample_id", "exp_num"])
    test_sampledf_resultdf = pd.DataFrame(test_sampledf_result)

    print("samplefilename:   ",test_sampledf.shape)
    print("submissionfilename:   ",test_sampledf_resultdf.shape)
    print('submission_data:   ')
    print(test_sampledf_result.head(10))
    test_sampledf = pd.merge(test_sampledf, test_sampledf_resultdf, how='left', left_on='sample_id', right_on='sample_id')
    test_sampledf.sort_values(by=["ad_id", "bid"], inplace=True)

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
    print("score ：" + str(60 * (monoscore + 1) / 2))

def win_ad(x):
    count = 0
    win_ads = []
    ads_list = x.split(';')
#     print(ads_list)
    for per_ad in ads_list:
        per_ad_attribute = per_ad.split(',')
        if(per_ad_attribute[-1] == '1' ):
            win_ads.append( ','.join(per_ad_attribute) )
    return '|'.join( win_ads )

def win_ad_count(x):
    count = 0
    ads_list = x.split(';')
#     print(ads_list)
    for per_ad in ads_list:
        per_ad_attribute = per_ad.split(',')
        if(per_ad_attribute[-1] == '1' ):
            count = count + 1
    return count


def extract_test_track_log_num_of_opponents(track_log_df):

    train_data = track_log_df.groupby(['request_day', 'location_id','ad_id']) \
        ['queue_length','request_id']. \
        agg({
        'queue_length': 'sum',
        'request_id': 'count'
    }).reset_index(). \
        rename(columns={
        'queue_length': 'num_of_opponents',
        'request_id': 'ad_id_request_num'
    })

    return train_data

def extract_test_ad_id_track_log_num_of_opponents(track_log_df):

    train_data = track_log_df.groupby(['request_day','ad_id']) \
        ['queue_length']. \
        agg({
        'queue_length': 'sum'
    }).reset_index(). \
        rename(columns={
        'queue_length': 'num_of_opponents',
    })

    return train_data

def my_train_group_data(df_code):

    number_of_opponents = sum(list(df_code['queue_length'].values))

    ad_id = df_code['ad_id'].iloc[0]
    request_day = df_code['request_day'].iloc[0]

    location_id_merge_temp = list(df_code['location_id'].values)
    one_day_location_id = ','.join(location_id_merge_temp)

    # request_id_merge_temp = list(df_code['request_id'].values)
    # one_day_request_id = ','.join(request_id_merge_temp)

    user_id_merge_temp = list(df_code['user_id'].values)
    one_day_user_id = ','.join(user_id_merge_temp)

    this_user_looked_ad_ids_merge_temp = list(df_code['this_user_looked_ad_ids'].values)
    one_day_this_user_looked_ad_ids = ','.join(this_user_looked_ad_ids_merge_temp)

    return_dict = {}
    compete_ad_id_merge_temp = list(df_code['compete_ad_id'].values)
    for val in compete_ad_id_merge_temp:
        return_dict.update(val)
    b = sorted(return_dict.items(), key=lambda item: item[0] , reverse= True)
    split_length = len( list(dict(b).values()) )
    split_length = min( split_length , 50 )
    one_day_compete_top_ad_id = list(dict(b).values())[0:split_length]
    one_day_compete_top_ad_id = ','.join([str(val) for val in one_day_compete_top_ad_id])

    compete_ad_bid_merge_temp = list(df_code['compete_ad_bid'].values)
    compete_ad_bid_merge_temp = ','.join(compete_ad_bid_merge_temp)
    compete_ad_bid_merge_temp = compete_ad_bid_merge_temp.split(',')
    compete_ad_bid_merge_temp = np.array(compete_ad_bid_merge_temp).flatten()
    try:
        compete_ad_bid_merge_temp = np.array([float(val) if (val != '') else 0 for val in compete_ad_bid_merge_temp])
        compete_ad_bid_merge_max = np.max(compete_ad_bid_merge_temp)
        compete_ad_bid_merge_min = np.min(compete_ad_bid_merge_temp)
        compete_ad_bid_merge_mean = np.mean(compete_ad_bid_merge_temp)
    except:
        compete_ad_bid_merge_max = 0
        compete_ad_bid_merge_min = 0
        compete_ad_bid_merge_mean = 0
        print(compete_ad_bid_merge_temp)

    compete_ad_pctr_merge_temp = list(df_code['compete_ad_pctr'].values)
    compete_ad_pctr_merge_temp = ','.join(compete_ad_pctr_merge_temp)
    compete_ad_pctr_merge_temp = compete_ad_pctr_merge_temp.split(',')
    compete_ad_pctr_merge_temp = np.array(compete_ad_pctr_merge_temp).flatten()
    try:
        compete_ad_pctr_merge_temp = np.array([float(val) if (val != '') else 0 for val in compete_ad_pctr_merge_temp])
        compete_ad_pctr_merge_max = np.max(compete_ad_pctr_merge_temp)
        compete_ad_pctr_merge_min = np.min(compete_ad_pctr_merge_temp)
        compete_ad_pctr_merge_mean = np.mean(compete_ad_pctr_merge_temp)
    except:
        compete_ad_pctr_merge_max = 0
        compete_ad_pctr_merge_min = 0
        compete_ad_pctr_merge_mean = 0
        print(compete_ad_pctr_merge_temp)

    compete_ad_quality_ecpm_merge_temp = list(df_code['compete_ad_quality_ecpm'].values)
    compete_ad_quality_ecpm_merge_temp = ','.join(compete_ad_quality_ecpm_merge_temp)
    compete_ad_quality_ecpm_merge_temp = compete_ad_quality_ecpm_merge_temp.split(',')
    compete_ad_quality_ecpm_merge_temp = np.array(compete_ad_quality_ecpm_merge_temp).flatten()
    try:
        # np.array([float(val) if (val != '') else 0 for val in compete_ad_quality_ecpm_merge_temp])
        compete_ad_quality_ecpm_merge_temp = np.array([float(val) if (val != '') else 0 for val in compete_ad_quality_ecpm_merge_temp])
        compete_ad_quality_ecpm_merge_max = np.max(compete_ad_quality_ecpm_merge_temp)
        compete_ad_quality_ecpm_merge_min = np.min(compete_ad_quality_ecpm_merge_temp)
        compete_ad_quality_ecpm_merge_mean = np.mean(compete_ad_quality_ecpm_merge_temp)
    except:
        compete_ad_quality_ecpm_merge_max = 0
        compete_ad_quality_ecpm_merge_min = 0
        compete_ad_quality_ecpm_merge_mean = 0
        print(compete_ad_pctr_merge_temp)

    compete_ad_total_ecpm_merge_temp = list(df_code['compete_ad_total_ecpm'].values)
    compete_ad_total_ecpm_merge_temp = ','.join(compete_ad_total_ecpm_merge_temp)
    compete_ad_total_ecpm_merge_temp = compete_ad_total_ecpm_merge_temp.split(',')
    compete_ad_total_ecpm_merge_temp = [val.split(',') for val in compete_ad_total_ecpm_merge_temp]
    compete_ad_total_ecpm_merge_temp = np.array(compete_ad_total_ecpm_merge_temp).flatten()
    try:
        compete_ad_total_ecpm_merge_temp = np.array([float(val) if (val != '') else 0 for val in compete_ad_total_ecpm_merge_temp])
        compete_ad_total_ecpm_merge_max = np.max(compete_ad_total_ecpm_merge_temp)
        compete_ad_total_ecpm_merge_min = np.min(compete_ad_total_ecpm_merge_temp)
        compete_ad_total_ecpm_merge_mean = np.mean(compete_ad_total_ecpm_merge_temp)
    except:
        compete_ad_total_ecpm_merge_max = 0
        compete_ad_total_ecpm_merge_min = 0
        compete_ad_total_ecpm_merge_mean =0
        print(compete_ad_total_ecpm_merge_temp)

    # policy_filter_flag_merge_temp = list(df_code['policy_filter_flag'].values)
    # one_day_policy_filter_flag = ','.join(policy_filter_flag_merge_temp)

    return ad_id, request_day,\
           number_of_opponents ,\
           one_day_location_id, \
           one_day_user_id, \
           one_day_this_user_looked_ad_ids,\
           one_day_compete_top_ad_id, \
           compete_ad_bid_merge_max, compete_ad_bid_merge_min, compete_ad_bid_merge_mean, \
           compete_ad_pctr_merge_max, compete_ad_pctr_merge_min, compete_ad_pctr_merge_mean, \
           compete_ad_quality_ecpm_merge_max, compete_ad_quality_ecpm_merge_min, compete_ad_quality_ecpm_merge_mean, \
           compete_ad_total_ecpm_merge_max, compete_ad_total_ecpm_merge_min, compete_ad_total_ecpm_merge_mean

def get_win_ad_attribute(x, index):
    per_ad = x['win_ad']
    per_ad_attribute = per_ad.split(',')
    return per_ad_attribute[index]

def plot_feature_importance(model):
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, max_num_features=30)
    plt.title("Featurertances")
    plt.show()

def plot_feature_importance_and_save_plt(model ,path):
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, max_num_features=30)
    plt.title("Featurertances")
    plt.savefig( path + "feature_importance.jpg")
    plt.show()
    # print(pd.DataFrame({
    #     'column': train_columns,
    #     'importance': gbm_1.feature_importance(),
    # }).sort_values(by='importance', ascending = False ))

def convert_time(x):
    x = str(x)
    return x[0:4] + '-' + x[4:6] + '-' + x[6:8] + ' ' + x[8:10] + ':' + x[10:12] + ':' + x[12:14]

def add_days(x,i):
    ss = datetime.strptime(x,'%Y-%m-%d') + timedelta(days = i)
    return ss.strftime('%Y-%m-%d')
#'%Y-%m-%d'

def create_sub_lagging(df,df_origin,i):
    df1 = df_origin.copy()
    df1['request_day'] = df1['request_day'].apply(add_days,args = (i,))
    df1 = df1.rename(columns = {'exp_num':'lagging' + str(i)})
    df2 = pd.merge(df,df1[['ad_id','request_day','lagging' + str(i)]],on = ['request_day','ad_id'],\
                  how = 'left')
    return df2

def create_competitive_lagging(df,df_origin,i, col_name):
    df1 = df_origin.copy()
    df1['request_day'] = df1['request_day'].apply(add_days,args = (i,))
    after_columns = 'lagging_'  + col_name + '_' + str(i)
    df1 = df1.rename(columns = {col_name:after_columns})
    # print(df1.columns)
    df2 = pd.merge(df,df1[['ad_id','request_day',after_columns]],on = ['request_day','ad_id'],\
                  how = 'left')
    return df2, after_columns

def create_col_list_rencent_competitive_lagging(df,df_origin,i, col_name,  col_list ):
    df1 = df_origin.copy()
    df1['request_day'] = df1['request_day'].apply(add_days,args = (i,))
    after_columns = '_'.join(col_list) + '_lagging_'  + col_name + '_' + str(i)
    df1 = df1.rename(columns = {col_name:after_columns})
    df1 = df1.groupby( col_list + ['request_day'] )[after_columns].mean().reset_index()
    # print(df1.columns)
    df2 = pd.merge(df,df1,on =  ['request_day'] + col_list ,  how = 'left')
    return df2, after_columns

def create_rencent_competitve_power_lagging(df,df_origin,i, target_col_name, col ):
    df1 = df_origin.copy()
    df1['request_day'] = df1['request_day'].apply(add_days,args = (i,))
    after_columns = col +'_lagging_' + '_'+ target_col_name + '_' + str(i)
    df1 = df1.rename(columns = {target_col_name:after_columns})
    try:
        df1 = df1.groupby([col,'request_day'])[after_columns].mean().reset_index()
    except:
        print(df1.columns)
    # print(df1.columns)
    df2 = pd.merge(df,df1[[ col ,'request_day',after_columns]],on = ['request_day', col ], how = 'left')
    return df2, after_columns

def create_new_lagging(data ):

    print("********** begin create_lagging begin******************")
    start = time.time()
    # lag
    data = create_sub_lagging(data, data, 1)
    for i in range(2, 2 + 2):
        data = create_sub_lagging(data, data, i)
    # lagging1_isnull_sum = data['lagging1'].isnull().sum()
    # lagging2_isnull_sum = data['lagging2'].isnull().sum()
    # lagging3_isnull_sum = data['lagging3'].isnull().sum()
    print(data['lagging1'].isnull().sum())
    print(data['lagging2'].isnull().sum())
    print(data['lagging3'].isnull().sum())

    data['lag_max'] = data.apply(get_new_Max, axis=1, args=('lagging1', 'lagging2', 'lagging3',))
    data['lag_min'] = data.apply(get_new_Min, axis=1, args=('lagging1', 'lagging2', 'lagging3',))
    data['lag_mean'] = data.apply(get_new_Mean, axis=1, args=('lagging1', 'lagging2', 'lagging3',))

    lagging_columns = ['lagging1', 'lagging2', 'lagging3', 'lag_max', 'lag_min', 'lag_mean']

    print("********** end create_lagging end******************")
    t = (time.time() - start) / 60
    print(data.shape)
    print("create_lagging running time: {:.2f} minutes\n".format(t))

    return data, lagging_columns

from math import isnan

def get_new_Max(row , lagging1 ,lagging2 ,lagging3):
    if( isnan(row[lagging1]) and isnan(row[lagging2])  and isnan(row[lagging3] )  == False):
        return row[lagging3]

    if(isnan(row[lagging1]) and isnan(row[lagging3]) and isnan(row[lagging2] )  == False ):
        return row[lagging2]

    if(isnan(row[lagging2]) and isnan(row[lagging3]) and isnan(row[lagging1] )  == False):
        return row[lagging1]


    if (isnan(row[lagging1]) and isnan(row[lagging2]) == False and isnan(row[lagging3]) == False):
        return max(row[lagging2] , row[lagging3])
    if (isnan(row[lagging2]) and isnan(row[lagging1]) == False and isnan(row[lagging3]) == False):
        return max(row[lagging1] , row[lagging2])
    if (isnan(row[lagging3]) and isnan(row[lagging1]) == False and isnan(row[lagging2]) == False):
        return max(row[lagging2] , row[lagging1])

    return max(row[lagging1] , row[lagging2] , row[lagging3])

def get_new_Min(row,  lagging1 ,lagging2 ,lagging3):
    if (isnan(row[lagging1]) and isnan(row[lagging2]) and isnan(row[lagging3]) == False):
        return row[lagging3]

    if (isnan(row[lagging1]) and isnan(row[lagging3]) and isnan(row[lagging2]) == False):
        return row[lagging2]

    if (isnan(row[lagging2]) and isnan(row[lagging3]) and isnan(row[lagging1]) == False):
        return row[lagging1]

    if (isnan(row[lagging1]) and isnan(row[lagging2]) == False and isnan(row[lagging3]) == False):
        return min(row[lagging2] , row[lagging3])
    if (isnan(row[lagging2]) and isnan(row[lagging1]) == False and isnan(row[lagging3]) == False):
        return min(row[lagging1] , row[lagging2])
    if (isnan(row[lagging3]) and isnan(row[lagging1]) == False and isnan(row[lagging2]) == False):
        return min(row[lagging2] , row[lagging1])


    return min(row[lagging1], row[lagging2], row[lagging3])

def get_new_Mean(row,  lagging1 ,lagging2 ,lagging3):
    # if(row['lagging1'].isnull() ):
    #     print(-1)
    # return (row['lagging1'] + row['lagging2']+ row['lagging3'])/3.0

    if( isnan(row[lagging1]) and isnan(row[lagging2])  and isnan(row[lagging3] )  == False):
        return row[lagging3]

    if(isnan(row[lagging1]) and isnan(row[lagging3]) and isnan(row[lagging2] )  == False ):
        return row[lagging2]

    if(isnan(row[lagging2]) and isnan(row[lagging3]) and isnan(row[lagging1] )  == False):
        return row[lagging1]

    if (isnan(row[lagging1]) and isnan(row[lagging2]) == False and isnan(row[lagging3]) == False):
        return (row[lagging2] + row[lagging3]) / 2.0
    if (isnan(row[lagging2]) and isnan(row[lagging1]) == False and isnan(row[lagging3]) == False):
        return (row[lagging1] + row[lagging2]) / 2.0
    if (isnan(row[lagging3]) and isnan(row[lagging1]) == False and isnan(row[lagging2]) == False):
        return (row[lagging2] + row[lagging1]) / 2.0

    return (row[lagging1] + row[lagging2] + row[lagging3]) / 3.0

def create_lagging(data):
    print("********** begin create_lagging begin******************")
    start = time.time()
    # lag
    data = create_sub_lagging(data, data, 1)
    for i in range(2, 2 + 2):
        data = create_sub_lagging(data, data, i)

    data['lag_max'] = data.apply(lambda row: getMax(row), axis=1)
    data['lag_min'] = data.apply(lambda row: getMin(row), axis=1)
    data['lag_mean'] = data.apply(lambda row: getMean(row), axis=1)
    # data['lag_median'] = data.apply(lambda row: getMedian(row), axis=1)
    data['lag_var'] = data.apply(lambda row: getVar(row), axis=1)

    # lagging_columns = ['lagging1', 'lagging2' , 'lagging3',
    #                    'lagging4', 'lagging5', 'lagging6',
    #                    'lagging7',
    #                     'lag_mean', 'lag_var']
    # data[empty_columns].fillna(-1, inplace=True)
    lagging_columns = ['lagging1', 'lagging2' , 'lagging3', 'lag_max','lag_min' , 'lag_mean']
    # for empty_col in lagging_columns:
    #     data[empty_col].fillna(0, inplace=True)

    # data[lagging_columns] = data[lagging_columns].fillna(0, inplace=True)
    print("********** end create_lagging end******************")
    t = (time.time() - start) / 60
    print()
    print("create_lagging running time: {:.2f} minutes\n".format(t))

    return data , lagging_columns

def getMin(row):
    return min(row['lagging1'], row['lagging2'], row['lagging3'])

def getMax(row):
    return max(row['lagging1'], row['lagging2'], row['lagging3'])


def getMean(row):
    return (row['lagging1'] + row['lagging2']+ row['lagging3'])/3.0

def getMedian(row):
    if(row['lagging1'] < row['lagging2'] and row['lagging1'] < row['lagging3'] ):
        return row['lagging1']
    if(row['lagging2'] < row['lagging1'] and row['lagging2'] < row['lagging3'] ):
        return row['lagging1']
    if(row['lagging3'] < row['lagging1'] and row['lagging3'] < row['lagging2'] ):
        return row['lagging1']

def getVar(row):
    return np.var(row['lagging1'] + row['lagging2']+ row['lagging3'])

def adjust_monotonic(x):
#     return  pd.Series(  x['exp_num'].values + math.log(x['ad_bid'].values - x['base_ad_bid'].values ) )
      if(x['ad_bid'] - x['base_ad_bid'] == 0):
            return x['exp_num']
      else:
            return x['exp_num'] + math.log(x['ad_bid'] - x['base_ad_bid'])

def adjust_monotonic_by_yu(x):
    return x['exp_num'] + 0.00001 * x['bid']

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def log_convert(y):
    return math.log(y + 1.0)

def exp_convert(x):
    return math.exp(x) - 1.0

def convertOneStr2Interval(x):
    x = int(x)
    bin_str = bin(x)[2:]
    bin_len = len(bin_str)
    r_pos = bin_str.rfind('1')
    if bin_len % 2 == 0:
        end_date = str(bin_len // 2) + ':00'
    else:
        end_date = str(bin_len // 2) + ':30'

    interval = bin_len - r_pos - 1
    if interval % 2 == 0:
        begin_date = str(interval // 2) + ':00'
    else:
        begin_date = str(interval // 2) + ':30'
    return begin_date + '-' + end_date

def convertStr2Interval(x):
    res_str = ''
    time_list = x.split(',')
    for time in time_list:
        res_str += convertOneStr2Interval(time) + ','
    return res_str[:-1]

def statis_feat(train, test, col, target_encoding):
    temp = train.groupby(col, as_index=False)[target_encoding].agg(
        {col + '_' + target_encoding + '_kfold_mean': 'mean', col + '_' + target_encoding + '_median': 'median'})
    #     temp[col+'_ctr'] = temp[col+'_click']/(temp[col+'_count']+3)
    test = pd.merge(test, temp, on=col, how='left')
    # test[col + '_' + target_encoding + '_mean'].fillna(test[col + '_' + target_encoding + '_mean'].median(),
    #                                                    inplace=True)
    # test[col + '_' + target_encoding + '_median'].fillna(test[col + '_' + target_encoding + '_median'].median(),
    #                                                      inplace=True)
    return test


def statis_ratio_of_feat(train, test, col_i, col_j, col_name):
    se = train.groupby([col_i, col_j])['cnt'].sum()
    dt = train[[col_i, col_j]]
    cnt = train[col_i].map(train[col_i].value_counts())
    temp = pd.merge(dt, se.reset_index(), how='left',
                    on=[col_i, col_j]).sort_index()
    temp.rename(columns={'cnt': col_name}, inplace=True)
    temp[col_name] = ((temp[col_name].fillna(value=0).values / cnt.values) * 100).astype(int)
    temp.drop_duplicates(subset=[col_i, col_j], inplace=True)

    test = pd.merge(test, temp, on=[col_i, col_j], how='left')
    del se, dt, cnt, temp
    gc.collect()

    return test

def statis_count_of_feat_col_list(train, test, col_i_list, col_j, col_name):

    se = train.groupby(col_i_list)[col_j].value_counts()
    se = pd.Series(1, index=se.index).sum(level=col_i_list)

    se = (se - se.min()) / (se.max() - se.min()) * 100
    temp = se.reset_index()
    temp.rename(columns={0: col_name }, inplace=True)
    temp[col_name] = temp[col_name].fillna(0).astype(int)
    temp.drop_duplicates(subset=[col_i_list], inplace=True)
    test = pd.merge(test, temp, on=[col_i_list], how='left')
    del se, temp
    gc.collect()

    return test

def statis_count_of_feat(train, test, col_i, col_j, col_name):

    se = train.groupby([col_i])[col_j].value_counts()
    se = pd.Series(1, index=se.index).sum(level=col_i)

    se = (se - se.min()) / (se.max() - se.min()) * 100
    temp = se.reset_index()
    temp.rename(columns={0: col_name }, inplace=True)
    temp[col_name] = temp[col_name].fillna(0).astype(int)
    temp.drop_duplicates(subset=[col_i], inplace=True)
    test = pd.merge(test, temp, on=[col_i], how='left')
    del se, temp
    gc.collect()

    return test

def saved_result(gbm, with_adjust_exp_num_df, path , return_columns):
    submissionfilename = path + "/submission.csv"
    with_adjust_exp_num_df.to_csv(submissionfilename, index=False, encoding="utf-8", header=None,
                                  columns=["sample_id", "adjusted_exp_num"])


def save_gbm_result(gbm, path,return_columns ):
    submissionfilename = path + "/submission.csv"
    gbm_importance_result = pd.DataFrame({
        'column': return_columns,
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance', ascending=False)
    # plot_feature_importance(gbm)
    gbm_importance_result.to_csv(path + "/feature_impoartance.csv", index=False)

def save_submission(history_new_ad_df):
    local_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    path = '../submission/' + local_time
    if not os.path.exists(path):
        os.mkdir(path)
    submissionfilename = path + "/submission.csv"
    history_new_ad_df.to_csv(submissionfilename, index=False, encoding="utf-8", header=None,
                                  columns=["sample_id", "adjusted_exp_num"])

    test_sampledf_result = pd.read_csv(submissionfilename,
                                       sep=",", header=None, names=["样本id", "曝光量"])
    print( np.mean(test_sampledf_result['曝光量'].values) )

    samplefilename = '../data/total_data/test_sample_bid.out'

    getMonoScore(samplefilename, submissionfilename)

def split_deliver_time(x):
    return  '，'.join(list(set(x.split(','))))

def get_duration(x):
    temp = x.split('，')
    max_duration = -1
    for val in temp:
        two = val.split('-')
        start = two[0]
        try:
            end = two[1]
        except:
            print(x)
            print(two)
        if (start.find('24') == 0):
            start = pd.to_datetime('23:59')
        else:
            start = pd.to_datetime(start)
        if (end.find('24') == 0):
            end = pd.to_datetime('23:59')
        else:
            try:
                end = pd.to_datetime(end)
            except:
                print(x)
                print(temp)
                print(val)
        max_duration = max(max_duration, (end - start).seconds)
    return max_duration

