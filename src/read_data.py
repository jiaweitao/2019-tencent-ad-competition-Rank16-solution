from utils import *

def read_test_track_log_df(test_track_log_path):
    names = ['request_id', 'request_time', 'user_id', 'location_id', 'bidding_Advertising_Information']

    track_log_df = pd.read_csv(test_track_log_path, delimiter='\t', \
                               parse_dates=['request_time'], header=None, names=names)

    del track_log_df['request_time']
    gc.collect()
    track_log_df['request_day'] = '2019-04-23'

    return track_log_df
def read_final_select_test_request_df(final_select_test_request_path ):
    names = ['ad_id', 'request_set']
    final_select_test_request_df = pd.read_csv(final_select_test_request_path, delimiter='\t', \
                                               header=None, names=names, dtype={
            "ad_id": str})
    return final_select_test_request_df

def read_test_sample_Data(simple_whole_test_path):
    test_sampleDF = pd.read_csv(simple_whole_test_path)
    print('test_sampleDF: ', test_sampleDF.shape)
    return test_sampleDF

def read_test_sample_bid_path_df(test_sample_bid_path ):
    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_sample_bid_path_df = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                          header=None, names=names, dtype={"sample_id": int
            , 'ad_id': str, "target_conversion_type": int, 'charge_type': int, "bid": int})
    return test_sample_bid_path_df

def read_sample_test_data(test_track_log_path ,final_select_test_request_path,test_sample_bid_path  ):
    names = ['request_id', 'request_time', 'user_id', 'location_id', 'bidding_Advertising_Information']

    test_track_log_df = pd.read_csv(test_track_log_path, delimiter='\t', \
                                    parse_dates=['request_time'], header=None, names=names,
                                    dtype={"request_id": int, 'user_id': object, "location_id": int}
                                    )

    del test_track_log_df['request_time']
    gc.collect()
    test_track_log_df['request_day'] = '2019-04-24'

    names = ['ad_id', 'request_set']
    final_select_test_request_df = pd.read_csv(final_select_test_request_path, delimiter='\t', \
                                               header=None, names=names, dtype={
                                        "ad_id": str})

    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_sample_bid_path_df = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                          header=None, names=names,dtype={"sample_id": int
            , 'ad_id': str, "target_conversion_type": int, 'charge_type': int, "bid": int })
    return test_track_log_df, final_select_test_request_df, test_sample_bid_path_df

def read_test_data(test_track_log_path ,final_select_test_request_path ,test_sample_bid_path ):
    names = ['request_id', 'request_time', 'user_id', 'location_id', 'bidding_Advertising_Information']

    test_track_log_df = pd.read_csv(test_track_log_path, delimiter='\t', \
                                    parse_dates=['request_time'], header=None, names=names,
                                    dtype={"request_id": int, 'user_id': object, "location_id": int}
                                    )

    del test_track_log_df['request_time']
    gc.collect()
    test_track_log_df['request_day'] = '2019-04-23'

    names = ['ad_id', 'request_set']
    final_select_test_request_df = pd.read_csv(final_select_test_request_path, delimiter='\t', \
                                               header=None, names=names, dtype={
                                        "ad_id": str})

    names = ['sample_id', 'ad_id', 'target_conversion_type', 'charge_type', 'bid']
    test_sample_bid_path_df = pd.read_csv(test_sample_bid_path, delimiter='\t', \
                                          header=None, names=names,dtype={"sample_id": int
            , 'ad_id': str, "target_conversion_type": int, 'charge_type': int, "bid": int })
    return test_track_log_df, final_select_test_request_df, test_sample_bid_path_df

def read_exposure_group_all_data(exposure_group_all_path):
    exposure_group_allDF =  pd.read_csv(exposure_group_all_path)

    return exposure_group_allDF

def read_group_train_all_data(group_train ):
    train_data =  pd.read_csv(group_train)

    return train_data

def read_ad_static_data(ad_static_path ):
    static_feature_names = ['ad_id', 'create_time', 'ad_acc_id', 'good_id', 'good_class', 'ad_trade_id', 'ad_size']
    ad_static_df = pd.read_csv(ad_static_path, delimiter='\t', \
                               parse_dates=['create_time'], header=None, names=static_feature_names,
                               dtype={'ad_id': int, "ad_acc_id": int, \
                                      "good_id": str, "good_class": str, "ad_trade_id": str, 'ad_size': str})
    ad_static_df['create_time'] = ad_static_df['create_time'].astype(int)
    ad_static_df['create_time'] = pd.to_datetime(ad_static_df['create_time'] + 8 * 3600, unit='s', utc=True)
    ad_static_df['create_time'] = ad_static_df['create_time'].dt.strftime('%Y-%m-%d')

    return ad_static_df

def read_ad_op_data(ad_operation_path ):
    operation_names = ['ad_id', 'create_update_time', 'op_type', 'target_conversion_type', 'charge_type', 'bid']
    ad_op_df = pd.read_csv(ad_operation_path, delimiter='\t', header=None, names=operation_names, \
                           dtype={"ad_id": int, 'create_update_time': int, "op_type": int, "op_set": int,
                                  "op_value": object})
    ad_op_df.loc[ad_op_df['create_update_time'] != 0, 'create_update_time'] = ad_op_df['create_update_time'].apply(
        convert_time)
    # ad_op_df.dropna(inplace = True )
    return ad_op_df

def read_ad_static_dynamic_merge_data(ad_static_dynamic_merge_path ):
    ad_static_dynamic_merge_df = pd.read_csv(ad_static_dynamic_merge_path ,  delimiter='\t')
    ad_static_dynamic_merge_df['create_datetime'] = pd.to_datetime(ad_static_dynamic_merge_df['create_time'], unit='s')
    # ad_static_dynamic_merge_df['create_datetime'] = ad_static_dynamic_merge_df['create_datetime'].astype(str)
    ad_static_dynamic_merge_df['create_datetime'] = ad_static_dynamic_merge_df['create_datetime'].dt.strftime('%Y-%m-%d')
    return ad_static_dynamic_merge_df


def read_user_data(userd_data_path):
    user_columns = ['age', 'gender', 'area', 'status',
                    'education', 'consuptionAbility', 'device', 'work', 'connectionType', 'behavior']
    user_df = pd.read_csv(userd_data_path, delimiter='\t', names = user_columns )

    return user_df

