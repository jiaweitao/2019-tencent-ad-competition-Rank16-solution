#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import logging
from src.utils import *
logging.basicConfig(level=logging.INFO)
from dataConfig import dataConfig

data_conf = dataConfig()
ad_operation_path = data_conf.ad_operation_path
ad_static_path = data_conf.ad_static_path

ad_op_mid_path = data_conf.ad_op_mid_path
ad_op_mid_simple_path = data_conf.ad_op_mid_simple_path
# ad_op_mid_simple_whole_path = '../../data/total_data/process_simple_data/ad_op_mid_simple_whole.txt'
ad_merge_path = data_conf.ad_merge_path


def handle_ad_op(ad_operation_path, ad_op_mid_path):

    if not os.path.exists(ad_operation_path):
        logging.info(ad_operation_path + ' not exits')
        return
    if os.path.exists(ad_op_mid_path):
        logging.info(ad_op_mid_path + ' already exits')
        return


    operation_names = ['ad_id', 'create_update_time', 'op_type', 'target_conversion_type', 'charge_type', 'bid']
    ad_op_df = pd.read_csv(ad_operation_path, delimiter='\t', header=None, names=operation_names, \
                           dtype={"ad_id": int, 'create_update_time': int, "op_type": int, "op_set": int,
                                  "op_value": object})

    logging.info(ad_operation_path + ' load success')


    def convert_time(x):
        x = str(x)
        return x[0:4] + '-' + x[4:6] + '-' + x[6:8] + ' ' + x[8:10] + ':' + x[10:12] + ':' + x[12:14]

    ad_op_df.loc[ad_op_df['create_update_time'] != 0, 'create_update_time'] = ad_op_df['create_update_time'].apply(
        convert_time)

    ad_op_df.sort_values(by=['ad_id', 'create_update_time'], inplace=True)
    ad_op_df = ad_op_df.reset_index()
    # ad_op_df.drop(columns = ['index'],inplace=True)
    ad_op_df.drop('index', axis=1, inplace=True)

    ad_op_df.to_csv(ad_op_mid_path, sep='\t', index=False)

    logging.info(ad_op_mid_path + ' dump success')

    del ad_op_df
    gc.collect()

def achieve_only_simple_ad_by_line(ad_op_mid_path, ad_op_mid_pathcreate_and_devised_path):
    if not os.path.exists(ad_op_mid_path):
        logging.info(ad_op_mid_path + ' not exits')
        return

    if os.path.exists(ad_op_mid_pathcreate_and_devised_path):
        logging.info(ad_op_mid_pathcreate_and_devised_path + ' already exits')
        return

    ad_op_df = pd.read_csv(ad_op_mid_path, delimiter='\t')
    # print(create_id_set)
    # print(devised_id_set)
    # print(only_create_id_set)
    # ad_id
    # create_update_time
    # op_type
    # target_conversion_type
    # charge_type
    # bid
    with open(ad_op_mid_pathcreate_and_devised_path, 'w') as w:
        w.write('ad_id\ttarget_conversion_type\tcharge_type\tbid\tvalid_start_time\n')
        # w.write('ad_id\tad_bid\tdeliver_time\ttarget_people\tvalid_start_time\n')

        # ad_id	create_update_time	op_type	op_set	op_value

        ad_id = ''
        target_conversion_type = None
        charge_type = ''
        bid = ''
        valid_start_time = ''

        index = 0

        logging.info('begin handle ad operation data')

        for print_index, row in ad_op_df.iterrows():
            if print_index % 10000 == 0:
                logging.info('read %d lines' % print_index)
            # logging.info(print_index)

            if row['op_type'] == 2:
                target_conversion_type = row['target_conversion_type']
                charge_type = row['charge_type']
                bid = row['bid']

                valid_start_time = row['create_update_time']
                ad_id = row['ad_id']

                w.write(str(ad_id) + '\t' + str(target_conversion_type) + '\t' + str(charge_type) + '\t' +
                            str(bid) + '\t' + str(
                        valid_start_time) + '\n')
                w.flush()
                continue

        logging.info('end handle ad operation data by line')
        del ad_op_df
        gc.collect()

def merge_only_create_ad_by_line(ad_op_mid_only_single_create_path, ad_op_mid_only_whole_create_path):
    if not os.path.exists(ad_op_mid_only_single_create_path):
        logging.info(ad_op_mid_only_single_create_path + ' not exits')
        return

    if os.path.exists(ad_op_mid_only_whole_create_path):
        logging.info(ad_op_mid_only_whole_create_path + ' already exits')
        return

    ad_op_df = pd.read_csv(ad_op_mid_only_single_create_path, delimiter='\t')

    def ab(ad_op_df):
        #     return pd.Series(list(set(ad_op_df.dropna())))
        return pd.Series(ad_op_df.values[-1])


    columns = ad_op_df.columns[1:]
    res = []
    for col in columns:
        res.append(ad_op_df.groupby('ad_id')[col].apply(ab))
    result = res[0]
    for i in range(1, len(res)):
        result = pd.concat([result, res[i]], axis=1)
    result = result.reset_index()
    result.drop('level_1', axis=1, inplace=True)

    result.to_csv(ad_op_mid_only_whole_create_path, sep='\t', index=0)

def merge(ad_static_path, ad_op_mid_path, ad_merge_path):
    if not os.path.exists(ad_static_path):
        logging.info(ad_static_path + ' not exits')
        return
    if not os.path.exists(ad_op_mid_path):
        logging.info(ad_op_mid_path + ' not exits')
        return
    if os.path.exists(ad_merge_path):
        logging.info(ad_merge_path + ' already exits')
        return

    # ad_op_df = pd.read_csv(ad_op_mid_path, delimiter='\t')

    ## static ad
    static_feature_names = ['ad_id', 'create_time', 'ad_acc_id', 'good_id', 'good_class', 'ad_trade_id', 'ad_size']
    # ad_static_df = pd.read_csv(ad_static_path,delimiter = '\t',\
    #                           parse_dates = ['create_time'],header=None,names = static_feature_names,dtype={'ad_id':object,"ad_acc_id": int,\
    #                           "good_id": str, "good_class": str, "ad_trade_id": str,'ad_size':str})
    ad_static_df = pd.read_csv(ad_static_path, delimiter='\t', \
                               parse_dates=['create_time'], header=None, names=static_feature_names,
                               dtype={'ad_id': int, "ad_acc_id": int, \
                                      "good_id": str, "good_class": str, "ad_trade_id": str, 'ad_size': str})

    # ad_static_df['ad_id'] = ad_static_df['ad_id'].astype(object)
    logging.info(ad_static_path + ' load success')

    new_op_df = pd.read_csv(ad_op_mid_path, delimiter='\t')
    new_op_df['ad_id'] = new_op_df['ad_id'].astype(object)
    # pd.DataFrame(columns = ['ad_id','ad_bid','deliver_time','target_people','valid_time'])
    logging.info(ad_op_mid_path + ' load success')



    ad_static_df.dropna(inplace=True)
    # ad_op_df.dropna(inplace=True)

    new_op_df = new_op_df.merge(ad_static_df, on='ad_id', how='left')
    # new_op_df.drop(columns = ['create_time'],inplace=True)
    # new_op_df.drop(['create_time'],axis = 1, inplace=True)
    logging.info('merge success')
    new_op_df.to_csv(ad_merge_path, sep='\t',index=0)
    logging.info(ad_merge_path + ' dump success')

def main():
    # ad_static_path = '../../data/testA/ad_static_feature.out'
    # ad_operation_path = '../../data/testA/ad_operation.dat'
    # ad_op_mid_path = '../../data/testA/process_simple_data/ad_op_mid.txt'
    # ad_op_mid_simple_path = '../../data/testA/process_simple_data/ad_op_mid_simple.txt'
    # ad_op_mid_simple_whole_path = '../../data/testA/process_simple_data/ad_op_mid_simple_whole.txt'
    # ad_merge_path = '../../data/testA/process_simple_data/ad_static_dynamic_merge.txt'


    handle_ad_op(ad_operation_path, ad_op_mid_path)
    achieve_only_simple_ad_by_line(ad_op_mid_path, ad_op_mid_simple_path)
    # merge_only_create_ad_by_line(ad_op_mid_simple_path, ad_op_mid_simple_whole_path)

    merge(ad_static_path, ad_op_mid_simple_path, ad_merge_path)

if __name__ == '__main__':
    main()
