# coding:utf-8
import os

class dataConfig:
    def __init__(self):

        self.train_track_log_path = '../data/total_data/track_log/'
        self.exposure_group_data_path = '../data/chusai_statstic_data/group_data/'
        self.exposure_group_all_path = '../data/chusai_statstic_data/exposure_group_all.csv'

        # self.group_train = './data/total_data/process_train_data/train_data.csv'
        # self.exposure_group_all_path = './data/whole_train_data/exposure_group_all.csv'


        self.ad_id_group_train_data_path = '../data/try_ad_id/group_train_data/'
        self.ad_id_train_path = '../data/try_ad_id/train_data.csv'

        self.location_id_group_train_data_path = '../data/try_location_id/group_train_data/'
        self.location_id_train_path = '../data/try_location_id/train_data.csv'


        self.test_track_log_path = '../data/total_data/BTest/BTest_tracklog_20190424.txt'
        self.final_select_test_request_path = '../data/total_data/BTest/Btest_select_request_20190424.out'
        self.test_sample_bid_path = '../data/total_data/BTest/Btest_sample_bid.out'

        self.with_dynamic_ad_id_whole_test_path = '../data/try_ad_id/final_select_test_request_df_with_track_df'
        self.simple_ad_id_whole_test_path = '../data/try_ad_id/simple_whole_test.txt'

        self.with_dynamic_location_id_whole_test_path = '../data/try_location_id/final_select_test_request_df_with_track_df'
        self.simple_location_id_whole_test_path = '../data/try_location_id/simple_whole_test.txt'


        self.ad_static_path = '../data/total_data/map_ad_static.out'
        self.ad_operation_path = '../data/total_data/final_map_bid_opt.out'
        # self.ad_static_dynamic_merge_path = '../data/total_data/process_simple_ad_data/ad_static_dynamic_merge.txt'


        self.ad_op_mid_path = '../data/whole_ad_data/ad_op_mid.txt'
        self.ad_op_mid_simple_path = '../data/whole_ad_data/ad_op_mid_simple.txt'
        # ad_op_mid_simple_whole_path = '../data/total_data/process_simple_data/ad_op_mid_simple_whole.txt'
        self.ad_merge_path = '../data/whole_ad_data/ad_static_dynamic_merge.txt'


        self.user_data_path = '../data/total_data/user_data.out'

        self.path = '../saved_features_data/'

        self.lgb_saved_file = self.path + 'saved_data' + '_lgb'
        self.lgb_return_numerical_columns_path = self.path + 'lgb_numerical_columns.csv'
        self.lgb_return_categorical_columns_path = self.path + 'lgb_categorical_columns.csv'

        self.xgb_saved_file = self.path + 'saved_data' + '_xgb'
        self.xgb_return_numerical_columns_path = self.path + 'xgb_numerical_columns.csv'
        self.xgb_return_categorical_columns_path = self.path + 'xgb_categorical_columns.csv'
























































