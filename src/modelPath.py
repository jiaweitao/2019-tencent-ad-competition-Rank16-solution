# coding:utf-8
import os

class modelPath:
    def __init__(self):
        #***********************************************
        # self.saved_feautres_path = '../saved_features_data/'
        # self.saved_file = self.saved_feautres_path + '_lgb_saved_data'
        # self.return_numerical_columns_path = self.saved_feautres_path + '_lgb_numerical_columns.csv'
        # self.return_categorical_columns_path = self.saved_feautres_path + '_lgb_categorical_columns.csv'


        # self.path = '../saved_features_data/'
        # guize
        self.submissionfilename = '../submission.csv'
        self.path = '../data/try_location_id/train_data.csv'
        self.simple_whole_test = '../data/try_location_id/simple_whole_test.txt'


        #lgb
        self.saved_get_recent_ecpm_path = '../saved_features_data/get_recent_ecpm/data.csv'
        self.get_recent_ecpm_path = '../saved_features_data/get_recent_ecpm/columns'
        self.Log_file = '../data/chusai_statstic_data/exposure_group_all.csv'
        self.test_sample_bid_path = '../data/total_data/BTest/Btest_sample_bid.out'
        self.samplefilename = '../data/total_data/BTest/Btest_sample_bid.out'

        #xgb

        # train  ad_id
        self.ad_id_train_path = '../data/try_ad_id/train_data.csv'
        # test   ad_id
        self.ad_id_test_path = '../data/try_ad_id/simple_whole_test.txt'

        # 处理过后的广告数据文件
        self.ad_op_mid_path = '../data/whole_ad_data/ad_op_mid.txt'
        self.ad_op_mid_simple_path = '../data/whole_ad_data/ad_op_mid_simple.txt'
        # ad_op_mid_simple_whole_path = '../../data/total_data/process_simple_data/ad_op_mid_simple_whole.txt'
        self.ad_merge_path = '../data/whole_ad_data/ad_static_dynamic_merge.txt'

        # load_clean_data
    #     my_load_clean_with_dynamic_data_path = '../data/try_location_id/train_data.csv'
    #
    # my_load_clean_with_dynamic_data_path = '../data/diff_ecpm/train_data.csv'

# my_load_clean_with_dynamic_data_path =     '../data/try/train_data.csv'










































