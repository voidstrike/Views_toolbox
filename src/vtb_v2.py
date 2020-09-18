"""A View toolbox that handles FILE.parquet & FILE.csv to corresponding D3M format features"""
import pandas as pd
import argparse
import os
import logging
import csv
import json

from typing import List
from create_dummies import createDummyProblemDoc, createDatasetDoc, createDataSplit
from auxiliary.utils import ConfigurationReader

# Auxiliary function that builds D3M folder strucutre -- Kind of ugly
def buildD3MRoot(tgt_path: str) -> None:
    logging.info('Start Creating D3M folders')
    if not os.path.exists(tgt_path):
        os.mkdir(tgt_path)

    for eachFolder in ['TRAIN', 'TEST', 'SCORE']:
        sub_path = os.path.join(tgt_path, eachFolder)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        for eachSubFolder in ['dataset', 'problem']:
            ssub_path = os.path.join(sub_path, f'{eachSubFolder}_{eachFolder}')
            if not os.path.exists(ssub_path):
                os.mkdir(ssub_path)
                if 'dataset' == eachSubFolder and not os.path.exists(os.path.join(ssub_path, 'tables')):
                    os.mkdir(os.path.join(ssub_path, 'tables'))
    
    logging.info('D3M folder creation complete')


def main(conf):
    # Read configuration and dataset

    print('Start Loading Dataset-----------------------------------------------------------------')
    dataset_configuration = ConfigurationReader(conf.config)
    x_list, y_list = dataset_configuration.feature_list, dataset_configuration.label_list
    # print(x_list)
    # print(y_list)
    # problem_doc = createDummyProblemDoc(dataset_configuration.data_name, [(y_list[0], len(x_list) + 1)], ['regression'], 'meanSquaredError')

    if dataset_configuration.file_type == 'csv':
        from container.CSVContainer import ViewContainer
        data_container = ViewContainer(dataset_configuration.file_path, dataset_configuration)
    elif dataset_configuration.file_type == 'parquet':
        from container.ParquetContainer import ViewContainer
        data_container = ViewContainer(dataset_configuration.file_path, dataset_configuration.engine)
    else:
        raise RuntimeError(f'[csv|parquet] is supposed to be given, but {dataset_configuration.file_type} shows up')

    print('Data Loading Finished-----------------------------------------------------------------')
    print('Start Manipulate Dataset---------------------------------------------------------------')

    # Manipulate the dataset
    data_container.filterByColumns_(x_list + y_list)

    if 'train' == dataset_configuration.mode:
        f_start, f_end = dataset_configuration.train_start, dataset_configuration.train_end
        s_start, s_end = dataset_configuration.val_start, dataset_configuration.val_end
    elif 'train_val' == dataset_configuration.mode:
        f_start, f_end = dataset_configuration.train_start, dataset_configuration.val_end
        s_start, s_end = dataset_configuration.test_start, dataset_configuration.test_end
    else:
        raise RuntimeError(f'Unsupported training mode is given {dataset_configuration.mode}')

    data_container.hard_manipulate(conf.test)
    data_container.filterByIndex(f_start, s_end)

    print('Data Manipulation Finished------------------------------------------------------------')
    print('Start Building D3M -------------------------------------------------------------------')
    # Start building D3M output
    d3m_folder = os.path.join(dataset_configuration.d3m_root, dataset_configuration.data_name)
    buildD3MRoot(d3m_folder)
    train_split_df, test_split_df = data_container.to_d3m(d3m_folder, f_start, f_end, s_start, s_end)

    # Finish remaining d3m file generation
    problem_doc = createDummyProblemDoc(dataset_configuration.data_name, [(y_list[0], len(x_list) + 1)], ['regression'], 'meanSquaredError')
    with open(os.path.join(d3m_folder, 'TRAIN/problem_TRAIN/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    with open(os.path.join(d3m_folder, 'TEST/problem_TEST/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    with open(os.path.join(d3m_folder, 'SCORE/problem_SCORE/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    
    tmp_list = train_split_df.columns.tolist()
    tmp_list = [item for item in tmp_list if item not in y_list]
    train_dataset_doc = createDatasetDoc(dataset_configuration.data_name, 'TRAIN', tmp_list, y_list)
    with open(os.path.join(d3m_folder, 'TRAIN/dataset_TRAIN/datasetDoc.json'), 'w') as outfile:
        json.dump(train_dataset_doc, outfile)

    test_dataset_doc = createDatasetDoc(dataset_configuration.data_name, 'TEST', tmp_list, y_list)
    with open(os.path.join(d3m_folder, 'TEST/dataset_TEST/datasetDoc.json'), 'w') as outfile:
        json.dump(test_dataset_doc, outfile)
    with open(os.path.join(d3m_folder, 'SCORE/dataset_SCORE/datasetDoc.json'), 'w') as outfile:
        json.dump(test_dataset_doc, outfile)

    train_split_doc = createDataSplit(train_split_df, mode='TRAIN')
    with open(os.path.join(d3m_folder, 'TRAIN/problem_TRAIN/dataSplits.csv'), 'w') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONE)
        for eachRow in train_split_doc:
            writer.writerow(eachRow)

    test_split_doc = createDataSplit(test_split_df, mode='TEST')
    with open(os.path.join(d3m_folder, 'TEST/problem_TEST/dataSplits.csv'), 'w') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONE)
        for eachRow in train_split_doc:
            writer.writerow(eachRow)

    with open(os.path.join(d3m_folder, 'SCORE/problem_SCORE/dataSplits.csv'), 'w') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONE)
        for eachRow in train_split_doc:
            writer.writerow(eachRow)
    pass


if __name__ == '__main__':
    # Parameters for this toolbox
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config/default.json', help='Path of the configuration')
    parser.add_argument('--test', action='store_true', help='Flag for "test"')

    parameters = parser.parse_args()
    main(parameters)

