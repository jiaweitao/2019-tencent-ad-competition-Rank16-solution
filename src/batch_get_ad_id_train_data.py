#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
import logging
import pandas as pd
import numpy as np
import pickle, os, jieba, time, gc, re
logging.basicConfig(level=logging.INFO)
import warnings

from dataConfig import dataConfig
warnings.filterwarnings("ignore")
from read_data import *

def try_new_group_data(file1_dir, file2_dir, group_file):

    if os.path.exists(group_file):
        logging.info(group_file + 'already exists')
        return

    dir_list = []
    read_dir = os.listdir(file1_dir)
    for file in read_dir:
        dir_list.append(file)

    write_file_list = ['exposure_group_' + str(i) + '.csv' for i in range(len(dir_list))]

    folds_dates = ['2019-04-10', '2019-04-11', '2019-04-12', '2019-04-13',
                   '2019-04-14', '2019-04-15', '2019-04-16', '2019-04-17',
                   '2019-04-18', '2019-04-19', '2019-04-20', '2019-04-21',
                   '2019-04-22']

    def handle_one_file(i, file_1, file_2):
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

        try:
            track_log_df['request_day'] = folds_dates[i]
        except:
            print(i)
            print(file1_dir)

        track_log_df['queue_length'] = track_log_df['bidding_Advertising_Information'] \
            .map(lambda x: len(x.split(';')) - 1.0 )

        # track_log_df['queue_length'] = track_log_df['temp_bidding_Advertising_Information'].map(
        #     lambda x: len(x.split(';')))
        # del track_log_df['temp_bidding_Advertising_Information']
        # gc.collect()
        def get_top_five(x):
            top_five = []
            ads_list = x.split(';')
            for per_ad in ads_list:
                per_ad_attribute = per_ad.split(',')
                try:
                    if (per_ad_attribute[5] == '1'):
                        continue
                except:
                    print('get_top_five')
                    print(per_ad_attribute)
                top_five.append(float(per_ad_attribute[4]))
            top_five = sorted(top_five)
            return np.mean(top_five[-5:])

        track_log_df['top_five_total_Ecpm_mean'] =  track_log_df['bidding_Advertising_Information'].map(get_top_five)

        track_log_df = (track_log_df.set_index(
          ['request_id', 'request_time', 'user_id', 'request_day','top_five_total_Ecpm_mean', 'queue_length'])[
                'bidding_Advertising_Information']
            .str.split(';', expand=True)
            .stack()
            .reset_index(level=6, drop=True)
            .reset_index(name='per_ad'))
        def get_per_ad_attribute_second(x):
            index = 1
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:
                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        def get_per_ad_attribute_thrid(x):
            index = 2
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        def get_per_ad_attribute_forth(x):
            index = 3
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        def get_per_ad_attribute_fifth(x):
            index = 4
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        def get_per_ad_attribute_sixth(x):
            index = 5
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        track_log_df['bidding_price'] = track_log_df['per_ad'].map(get_per_ad_attribute_second)
        track_log_df['pctr'] = track_log_df['per_ad'].map(get_per_ad_attribute_thrid)
        track_log_df['quality_ecpm'] = track_log_df['per_ad'].map(get_per_ad_attribute_forth)
        track_log_df['totalEcpm'] = track_log_df['per_ad'].map(get_per_ad_attribute_fifth)
        track_log_df['policy_filter'] = track_log_df['per_ad'].map(get_per_ad_attribute_sixth)
        def get_per_ad_attribute_first(x):
            index = 0
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        def get_per_ad_attribute_seventh(x):
            index = 6
            per_ad = x
            per_ad_attribute = per_ad.split(',')
            try:

                per_ad_attribute[index]
            except:
                print(index)
                print(per_ad_attribute)
                return 0
            return per_ad_attribute[index]
        track_log_df['ad_id'] = track_log_df['per_ad'].map(get_per_ad_attribute_first)
        track_log_df['exp_num'] = track_log_df['per_ad'].map(get_per_ad_attribute_seventh)


        columns = ['ad_id','exp_num' ]
        for col in columns:
            track_log_df[col] = track_log_df[col].astype(int, inplace = True )
        columns = ['bidding_price', 'pctr','quality_ecpm','totalEcpm']
        for col in columns:
            track_log_df[col] = track_log_df[col].astype(float, inplace=True)

        del track_log_df['per_ad']
        gc.collect()

        statstic_option = 'mean'
        track_log_df['ad_id_request_num'] = track_log_df['exp_num']
        group_df = track_log_df.groupby(['request_day', 'ad_id']) \
            ['exp_num', 'ad_id_request_num', 'bidding_price',
             'pctr','quality_ecpm', 'totalEcpm','queue_length']. \
            agg({'exp_num': 'sum',
                 'ad_id_request_num': 'count',
                 'bidding_price': statstic_option, \
                 'pctr': statstic_option, \
                 'quality_ecpm': statstic_option, \
                 'totalEcpm': statstic_option,
                 'queue_length': 'sum'
                 }).reset_index(). \
            rename(columns={
            'bidding_price': 'bidding_price_' + statstic_option,
            'pctr': 'pctr_' + statstic_option,
            'quality_ecpm': 'quality_ecpm_' + statstic_option,
            'totalEcpm': 'totalEcpm_' + statstic_option,
            'queue_length': 'num_of_opponents'
        })


        # group_df = pd.merge(group_df, ad_id_one_day_request_num_df, on='ad_id', how='left')
        group_df['exp_num'] = group_df['exp_num'].astype(float)
        group_df['ad_id_request_num'] = group_df['ad_id_request_num'].astype(float)
        group_df['ad_id_winning_probability'] = group_df['exp_num'].values / group_df['ad_id_request_num'].values

        group_df['num_of_opponents'] = group_df['num_of_opponents'].astype(float)
        group_df['per_times_winning_rate'] = group_df['exp_num'].values / group_df['num_of_opponents'].values

        # group_df['strong_person_winning_rate_in_one_day_request_num'] = group_df['high_quality_scene_nums'].values / group_df['ad_id_request_num'].values

        # group_df['strong_person_winning_rate_in_one_day_exp_num'] = group_df['high_quality_scene_nums'].values / group_df['exp_num'].values

        logging.info('group data success')
        del track_log_df
        gc.collect()

        train_data = group_df
        # train_data = pd.merge(train_data , group_df , on = ['ad_id','request_day'] , how = 'left')
        train_data.sort_values(by = ['request_day','ad_id'], inplace = True)
        print('*********** end **************')

        train_data.to_csv(file_2, header=True, index=None, sep=';', mode='w')
        del train_data
        gc.collect()

        logging.info(file_2 + ' dump success')

    for i, read_file in enumerate(dir_list):
        handle_one_file(i, file1_dir + read_file, file2_dir + write_file_list[i])

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
    df1.to_csv(group_file, index=None, sep='\t')
    logging.info(group_file + ' dump success')

if __name__ == '__main__':
    data_conf = dataConfig()

    train_track_log_path = data_conf.train_track_log_path
    ad_id_group_train_data_path = data_conf.ad_id_group_train_data_path
    ad_id_train_path =  data_conf.ad_id_train_path

    try_new_group_data(train_track_log_path, ad_id_group_train_data_path, ad_id_train_path)
