'''

(your_py_env) $ python speed_test.py dai_url dai_user dai_password metadata.csv
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


def get_dai_client(daiaddress, daipassword) -> 'driverlessai._core.Client':
    '''
    DAIサーバへの接続
    ----------
    daiaddress : str
    daipassword : str
    '''
    print('----- start server connection : get_dai_client -----')
    # Driverless AIサーバーへの接続
    dai = driverlessai.Client(address=daiaddress, username=dai_user, password=daipassword)
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

def get_experiment(daiobj, dataobj, target_column, task, drop_columns, enable_gpus) -> 'driverlessai._experiments.Experiment':
    '''
    Experimentの実行とExperimentオブジェクトの取得
    ----------
    daiobj : driverlessai._core.Client
    dataobj : driverlessai._datasets.Dataset
    target_column : str
    task : str
    drop_columns : List[str]
    enable_gpus : bool
    '''
    print('----- start experiment : get_experiment -----')
    # Experiment設定
    dai_settings = {
        'train_dataset': dataobj, 
        'target_column': target_column,
        'task': task,
        'drop_columns': drop_columns,
        'enable_gpus': enable_gpus
    }
    # Experimentの実行
    ex = daiobj.experiments.create(**dai_settings)
    return ex


def main():
    # Experiment Meta
    df_expperiments_info = pd.read_csv(expperiments_info)
    #print(df_expperiments_info.dtypes)

    for _, row in df_expperiments_info.iterrows():      # Experiment Metaの行でループ

        start_time = datetime.datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')   # 開始時間
        data_name = row['data_name']  # 実験データ
        print('#####-----  開始時間: ', start_time, '  -----#####')
        print('#####-----  実験データ: ', data_name, '  -----#####')

        for exp_try in range(row['try_n']):    # 同じ実験データでTry回数実施
            print('#####-----  Try: ', exp_try, '  -----#####')
            s3url = row['s3url']  # DAIにアップされてない場合の取得先S3
            # Experiment設定
            target_column = row['target_column']
            task = row['task']    # 'regression', 'classification', or 'unsupervised'
            if row['drop_columns']  is np.nan:     # dropped clmを指定しない場合
                drop_columns = []
            else:
                drop_columns = row['drop_columns'] .split(',')     # strをList化
            #print(drop_columns)
            enable_gpus = row['enable_gpus']
            print('GPU設定: {}'.format(enable_gpus))


            print('*************** DAIへ接続 ***************')
            dai = get_dai_client(daiaddress=dai_address, daipassword=dai_password)
            print('DAIバージョン: {}'.format(dai.server.version))


            print('*************** データの取得 ***************')
            ds = get_dataset(daiobj=dai, dataname=data_name, dataurl=s3url) 

            print('Dataサイズ(byte): {}'.format(ds.file_size))
            print('Dataサイズ(mega byte): {}'.format(ds.file_size/1024**2))
            print('Data shape: {}'.format(ds.shape))


            print('*************** Experimentの実施 ***************')
            ex = get_experiment(daiobj=dai, dataobj=ds, 
                                target_column=target_column, task=task, drop_columns=drop_columns, enable_gpus=enable_gpus)

            print('学習時間（sec）：{}'.format(ex.run_duration))
            print('学習時間（min）：{}'.format(ex.run_duration/60))
            print('Experimentサイズ（byte）：{}'.format(ex.size))
            print('Experimentサイズ（mega byte）：{}'.format(ex.size/1024**2))
            print('********** Experiment Summary **********')
            print(ex.summary())
            
            save_dict = dict(Meta_Data=expperiments_info,
                    Data_Name=data_name,
                    Try=row['try_n'],
                    Datasize_mb = ds.file_size/1024**2,
                    N_Observation = ds.shape[0],
                    N_features = ds.shape[1] - len(drop_columns) - 1,
                    Duration_min = ex.run_duration/60,
                    Experiment_Size_mb = ex.size/1024**2,
                    Acc_Time_Interpret = (ex.settings['accuracy'], ex.settings['time'], ex.settings['interpretability'])
                    )
            save_file = os.path.join('result', 'speedtest_{}.json'.format(start_time))
            with open(save_file, 'w') as f:
                json.dump(save_dict, f, indent=4)


if __name__ == '__main__':
    main()
