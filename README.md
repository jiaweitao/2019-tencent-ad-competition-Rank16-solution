个人博客地址： https://www.cnblogs.com/xianbin7/p/11076009.html
知乎地址：     https://zhuanlan.zhihu.com/p/71609590
这是复赛代码
文件目录结构：
    ./data下有五个目录：
    ./data/chusai_statstic_data
    ./data/total_data
    ./data/try_ad_id
    ./data/try_location_id
    ./data/whole_ad_data

    原始数据放置位置：
      在./data/total_data目录下放置：
           1  atest_tracklog.zip
           2  track_log.zip
           3  user_data.zip
           4  final_map_bid_opt.out
           5  map_ad_static.out
    最终文件目录如下：
       原始测试集的三部分文件： ./data/total_data/BTest下 (Btest_sample_bid.out,Btest_select_request_20190424.out,BTest_tracklog_20190424.txt)cd ..
       原始的分天历史日志数据文件： ./data/total_data/track_log
       原始广告静态数据：   ./data/total_data/map_ad_static.out
       原始广告操作数据： ./data/total_data/final_map_bid_opt.out 下放置


       分批处理的统计文件： ./data/chusai_statstic_data/group_data/ 下放置
       合并后的统计文件： ./data/chusai_statstic_data/group_data/exposure_group_all.csv
       训练集与测试集文件路径
          建模方式一 train  直接求广告id一天的曝光量
          './data/try_ad_id/group_train_data/'
          './data/try_ad_id/train_data.csv'
          './data/try_ad_id/simple_whole_test.txt'
          # 建模方式二 train  求广告id，一天内在各个广告位上的曝光量， 最后聚合得到广告id一天内的曝光量
          './data/try_location_id/group_train_data/'
          './data/try_location_id/train_data.csv'
          './data/try_location_id/simple_whole_test.txt'
       处理过后的广告数据文件：./whole_ad_data

       产生的xgb与lgb带有中间特征的训练集  ./saved_features_data
       中间结果文件 ：  ./submission/目录
       最终结果文件:  ./submission.csv
    由于数据量过大，我采样了历史日志文件作为原始输入
环境配置：
    Anaconda 4.5.13, Python 3.6.2

主要依赖库：
    pandas 0.20.3,
    numpy 1.13.1,
    scikit-learn 0.19.0,
    lightgbm_version 2.2.2,
    xgboost_version 0.80

步骤说明：

    按run.sh顺序运行代码，先后完成预处理、特征工程、模型训练、测试集预测与模型结果融合

预处理：
    建模方式1：
        分批处理每天的历史日志文件得到每天的广告id的曝光量，通过每行样本的历史竞价队列提取出曝光样本与
        未曝光样本，统计广告id，在当前天对应的请求次数，竞争对手的个数，竞价均值，pctr均值，quality_ecpm
        均值，total_ecpm均值，广告id在当前天的获胜概率(曝光次数除以请求次数)，广告id在当前天每次的获胜概率
        (曝光次数除以竞争对手的个数)
    建模方式2：
        与建模方式1类似，只不过每天广告id的曝光量替换成每天，广告id，在某个广告位的曝光量。

特征提取：

    四大类特征，历史平移特征、五折统计特征、时间特征、预估曝光量(历史平移乘以广告id当天的请求次数或者广告id的竞争对手个数)
    final_lgb.py使用特征：
    1  get_recent_ecpm()/create_new_lagging产生广告历史平移曝光量
    2  get_recent_ecpm()产生广告历史平移的pctr，quality_ecpm,toal_ecpm
    3  get_recent_fail_num()产生广告的请求失败次数
    4  generate_data_log_competitive_and_user_cover_feature()利用五折交叉统计产生广告素材下不同广告位的数目
    5  get_kflod_statstic_data()利用五折交叉统计广告的请求失败次数，曝光量，以及获胜概率
    6  get_rencent_winning_probability_exp_num()利用历史平移统计广告的曝光量以及获胜概率
    7  get_excepted_exp_num_columns()利用广告的历史平移的获胜概率均值乘以当天的请求次数
    8  get_delata()将请求时间，创建时间，请求时间减去创建时间作为特征
    9  对类别型特征进行ont-hot

    final_xgb.py使用特征：
    1  get_recent_ecpm()/create_new_lagging产生广告历史平移曝光量
    2  get_recent_ecpm()产生广告历史平移的pctr，quality_ecpm,toal_ecpm
    3  generate_data_log_competitive_and_user_cover_feature()利用五折交叉统计产生广告素材下不同广告位的数目
    4  get_rencent_per_times_winning_probability()利用历史平移得到广告当前天预估的单次获胜概率
    5  get_per_times_winning_rate_estimated_exp_num_columns利用当前天预估的单次获胜概率乘以当天的竞争对手的个数
    6  get_rencent_winning_probability_exp_num()利用历史平移统计广告的曝光量以及获胜概率
    7  get_delata()将请求时间，创建时间，请求时间减去创建时间作为特征
    8  对类别型特征进行ont-hot

模型融合：
    新旧广告定义： 未在训练集中出现的广告id为历史旧广告，否则为新广告，历史旧广告采用规则，新广告采用模型
    规则融合：规则一采用广告id在某个广告位的历史获胜概率乘以广告id当天在这个广告位的请求次数，规则二采用
             广告id在某个广告位的单次历史获胜概率乘以广告id当天在这个广告位竞争对手的个数， 采用加权融合
    模型融合：模型一使用lightgbm训练，模型二使用xgboost训练，特征方面的差异主要有lightgbm使用广告id在
            某个广告位的历史获胜概率以及广告id当天在这个广告位的请求次数构造特征，模型二采用广告id在某个
            广告位的单次历史获胜概率以及广告id当天在这个广告位竞争对手的个数构造特征， 最终采用加权融合

