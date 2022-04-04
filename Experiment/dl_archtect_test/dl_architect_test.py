'''
(your_py_env) $ python dl_architect_test.py dai_url dai_user dai_password experiment_meta_data/exp_params_car.csv

result is storing in ./result
'''

import os
import sys
import json
import datetime

import pandas as pd
import numpy as np

import driverlessai
print('driverlessaiパッケージバージョン： {}'.format(driverlessai.__version__))

# Driverless AIサーバー情報
dai_address = sys.argv[1]  # like 'http://3.82.142.224'
dai_user = sys.argv[2]
dai_password = sys.argv[3]
print('DAI URL： {}'.format(dai_address))
print('DAI User： {}'.format(dai_user))

# テストするExperiment情報（Experiment Meta）
expperiments_info = sys.argv[4]   # like 'experiment_meta_data/Experiments_Params.csv'
print('Experiments Meta： {}'.format(expperiments_info))


def get_dai_client(daiaddress, daiuser, daipassword) -> 'driverlessai._core.Client':
    '''
    DAIサーバへの接続
    ----------
    daiaddress : str
    daipassword : str
    '''
    print('----- start server connection : get_dai_client -----')
    # Driverless AIサーバーへの接続
    dai = driverlessai.Client(address=daiaddress, username=daiuser, password=daipassword)
    return dai

def get_dataset(daiobj, dataname, dataurl) -> 'driverlessai._datasets.Dataset': 
    '''
    データオブジェクトの取得
    ----------
    daiobj : driverlessai._core.Client
    dataname : str
    dataurl : str
    '''
    print('----- start get data : get_dataset -----')
    # DAI上のデータ一覧
    uploaded_data = {i.name:i.key for i in daiobj.datasets.list()}
    print('Uploaded data name : key >> ', uploaded_data)

    # データ取得
    if dataname in uploaded_data.keys():
        print('Data is already uploaded in DAI')
        ds = daiobj.datasets.get(uploaded_data[dataname]) 
    else:
        print('Data is uploading to DAI.')
        ds = daiobj.datasets.create(data=dataurl, data_source='s3')
    
    return ds

def get_experiment(daiobj, splitdata, target_column, task, drop_columns, tensorflow_image_pretrained_models, test_mode)-> 'driverlessai._experiments.Experiment':
    '''
    Experimentの実行とExperimentオブジェクトの取得
    ----------
    daiobj : driverlessai._core.Client
    dataobj : driverlessai._datasets.Dataset
    target_column : str
    task : str
    drop_columns : List[str]
    '''
    print('----- start experiment : get_experiment -----')
    # Experiment設定    
    dai_settings = {
        'target_column': target_column,
        'task': task,
        'drop_columns': drop_columns,
        'tensorflow_image_pretrained_models':[tensorflow_image_pretrained_models]
    }
    if test_mode:    # test modeの時、Acc=1&Time=1のExperimentを実施
        dai_settings['accuracy'] = 1
        dai_settings['time'] = 1
    
    # Experimentの実行
    ex = daiobj.experiments.create(**splitdata, **dai_settings)
    return ex


def run_whole_experiments(dai_address: str, dai_user: str, dai_password: str, df_expperiments_info: pd.DataFrame) -> None:

    # データ情報
    data_name = df_expperiments_info['data_name'][0]
    s3url = df_expperiments_info['s3url'][0]  # DAIにアップされてない場合の取得先S3
    print('#####-----  利用データ: ', data_name, '  -----#####')

    print('*************** DAIへ接続 ***************')
    dai = get_dai_client(daiaddress=dai_address, daiuser=dai_user, daipassword=dai_password)
    print(type(dai))
    print('DAIバージョン: {}'.format(dai.server.version))

    print('*************** データの取得 ***************')
    ds = get_dataset(daiobj=dai, dataname=data_name, dataurl=s3url) 

    print(type(ds))
    print('Dataサイズ(byte): {}'.format(ds.file_size))
    print('Dataサイズ(mega byte): {}'.format(ds.file_size/1024**2))
    print('Data shape: {}'.format(ds.shape))
        
    print('*************** データ分割 ***************')
    ds_split = ds.split_to_train_test(train_size=0.75, train_name=data_name+'_train', test_name=data_name+'_test')
    
    
    for _, row in df_expperiments_info.iterrows():
        #**********  実験のパラメータ情報  **********#
        tensorflow_image_pretrained_models = row['tensorflow_image_pretrained_models']
        start_time = datetime.datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')
        print('#####-----  開始時間: ', start_time, '  -----#####')
        print('#####-----  archtect: ', tensorflow_image_pretrained_models, '  -----#####')
        

        # Experiment設定
        target_column = row['target_column']
        task = row['task']    # 'regression', 'classification', or 'unsupervised'
        if row['drop_columns']  is np.nan:     # dropped clmを指定しない場合
            drop_columns = []
        else:
            drop_columns = row['drop_columns'] .split(',')     # strをList化
        #print(drop_columns)
        test_mode = row['test_mode']
        

        print('*************** Experimentの実施 ***************')
        ex = get_experiment(daiobj=dai, splitdata=ds_split, 
                            target_column=target_column, task=task, drop_columns=drop_columns, 
                           tensorflow_image_pretrained_models=tensorflow_image_pretrained_models, test_mode=test_mode)

        print(type(ds))
        print('学習時間（sec）：{}'.format(ex.run_duration))
        print('学習時間（min）：{}'.format(ex.run_duration/60))
        print('Experimentサイズ（byte）：{}'.format(ex.size))
        print('Experimentサイズ（mega byte）：{}'.format(ex.size/1024**2))
        print('精度：{}'.format(ex.metrics()))
        print('********** Experiment Summary **********')
        ex.summary()
        
        save_dict = dict(Data_Name=data_name,
                         Try=row['try_n'],
                         Datasize_mb = ds.file_size/1024**2,
                         N_Observation = ds.shape[0],
                         N_features = ds.shape[1] - len(drop_columns) - 1,
                         Shape_Train = ds_split['train_dataset'].shape,
                         Shape_Test = ds_split['test_dataset'].shape,
                         Network = tensorflow_image_pretrained_models,
                         Duration_min = ex.run_duration/60,
                         Experiment_Size_mb = ex.size/1024**2,
                         Acc_Time_Interpret = (ex.settings['accuracy'], ex.settings['time'], ex.settings['interpretability']),
                         Metrics = ex.metrics()
                        )
        save_file = os.path.join('result', 'speedtest_{}.json'.format(start_time))
        with open(save_file, 'w') as f:
            json.dump(save_dict, f, indent=4)


def main():
    # Experiment Meta
    df_expperiments_info = pd.read_csv(expperiments_info)
    #print(df_expperiments_info.dtypes)
    run_whole_experiments(dai_address=dai_address, dai_user=dai_user, dai_password=dai_password, df_expperiments_info=df_expperiments_info)


if __name__ == '__main__':
    main()
