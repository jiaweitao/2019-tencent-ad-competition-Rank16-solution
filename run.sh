# 一  复赛数据预处理
# --------在data/try_location_id目录下生成求广告id在某个广告位下一天的曝光量的训练集与测试集----------------------
echo ---------------data/try_location_id----------------------
python ./src/batch_get_ad_id_location_id_test_data.py
python ./src/batch_get_ad_id_location_id_train_data.py
# ------------在data/try_ad_id目录下生成求广告id一天的曝光量的训练集与测试集----------------
echo ------------data/try_ad_id----------------
python ./src/batch_get_ad_id_test_data.py
python ./src/batch_get_ad_id_train_data.py
#------------在data/batch_process_exposure_back/目录下产生广告尺寸下有多少个广告位的特征统计文件----------------------
echo------------data/batch_process_exposure_back----------------------
python ./src/batch_process_exposure_back.py
# ------产生处理过后的广告数据文件，带有完整的动态广告属性与静态广告属性----------
echo ------------data/whole_ad_data----------------
python ./src/process_ad_data_create_info.py
# 二  预测部分
# lightgbm 部分
echo --------------------------------lightgbm------------------------------------
python ./src/final_lgb.py
# xgboost 部分
echo --------------------------------xgboost------------------------------------
python ./src/final_xgb.py
# 最终融合
echo ---------------------------------------blending--------------------------------------------
python ./src/final_merge.py
