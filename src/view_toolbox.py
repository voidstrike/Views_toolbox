"""A View toolbox that handles FILE.parquet to corresponding D3M format features"""
import pandas as pd
import argparse
import os
import logging
import csv
import json

from typing import List
from src.create_dummies import createDummyProblemDoc, createDatasetDoc, createDataSplit

# GLOBAL SETTING BLOCK

# PGD FEATURES
# Borrowed from Yifan, 'ged_best_sb' is removed
PGD_X = ['acled_count_pr', 'acled_fat_pr', 'ged_best_ns', 'ged_best_os', 'ged_count_os', 'ged_count_sb', 'ged_count_ns',
        'pgd_agri_ih', 'pgd_aquaveg_gc', 'pgd_barren_ih', 'pgd_bdist3', 'pgd_capdist', 'pgd_cmr_mean', 'pgd_diamprim', 'pgd_drug_y',
        'pgd_excluded', 'pgd_forest_gc', 'pgd_forest_ih', 'pgd_gcp_mer', 'pgd_goldplacer', 'pgd_goldsurface', 'pgd_goldvein',
        'pgd_grass_ih', 'pgd_gwarea', 'pgd_herb_gc', 'pgd_imr_mean', 'pgd_landarea', 'pgd_maincrop', 'pgd_mountains_mean',
        'pgd_pasture_ih', 'pgd_petroleum', 'pgd_pop_gpw_sum', 'pgd_savanna_ih', 'pgd_shrub_gc', 'pgd_temp', 'pgd_ttime_mean',
        'pgd_urban_gc', 'pgd_urban_ih', 'pgd_water_gc', 'pgd_agri_gc', 'pgd_barren_gc', 'pgd_diamsec', 'pgd_gem', 'pgd_harvarea',
        'pgd_nlights_calib_mean', 'pgd_shrub_ih', 'pgd_water_ih']
PGD_Y = ['ln_ged_best_sb']

# CM FEATURES -- PENDING
CM_X = None
CM_Y = ['ln_ged_best_sb']

# A Container class for the given parquet
class DataContainer(object):
    def __init__(self, data_path: str, engine: str) -> None:
        self.path = data_path
        self.data = pd.read_parquet(self.path, engine=engine)
        self.df_idx = pd.IndexSlice

    def filterByColumns(self, feature_list: List[str]) -> None:
        self.data = self.data.filter(items=feature_list)
    
    def filterByIndex(self, start_idx: int, end_idx: int) -> None:
        # HARD CODE : FIX Later
        self.data = self.data.reorder_levels(['pg_id', 'month_id']).sort_index()
        self.data.ln_ged_best_sb = self.data.ln_ged_best_sb.shift(-1)
        self.data = self.data.reorder_levels(['month_id', 'pg_id']).sort_index()
        self.data = self.data.loc[self.df_idx[start_idx:end_idx]]

    def generateCSV(self, d3m_root, train_start, train_end, test_start, test_end):
        train_split = self.data.loc[self.df_idx[train_start:train_end]]
        train_len = len(train_split.index)

        tmp_df = self.data.reset_index(drop=True)
        tmp_df.index.name = 'd3mIndex'

        train_split = tmp_df.loc[:train_len]
        test_split = tmp_df.loc[train_len:]

        # SAVING
        training_file_path = os.path.join(d3m_root, 'TRAIN/dataset_TRAIN/tables/learningData.csv')
        train_split.to_csv(training_file_path)
        test_file_path = os.path.join(d3m_root, 'TEST/dataset_TEST/tables/learningData.csv')
        test_split.to_csv(test_file_path)
        test_file_path = os.path.join(d3m_root, 'SCORE/dataset_SCORE/tables/learningData.csv')
        test_split.to_csv(test_file_path)

        return train_split, test_split

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


def createSplit(train_idx, val_idx, test_idx, mode='bottom') -> None:
    # BOTTOM MODE : Combine train and val split, leave 'test'
    # TRAIN MODE : Ignore test_idx
    # RANDOM MODE : Combine all three splits and random split into train/test
    if 'bottom' == mode:
        return train_idx[0], val_idx[1], test_idx[0], test_idx[1]
    elif 'train' == mode:
        return train_idx[0], train_idx[1], val_idx[0], val_idx[1]
    else:
        # TODO
        raise NotImplementedError(f'SPLIT MODE {mode} is not supported yet!!!')
            
def main(conf):
    data_container = DataContainer(conf.parquet, conf.engine)
    if 'pgd' == conf.mode:
        x_list, y_list = PGD_X, PGD_Y
    elif 'cm' == conf.mode:
        x_list, y_list = CM_X, CM_Y
    else:
        raise NotImplementedError(f'Unsupported mode: {conf.mode}')

    full_list = x_list + y_list
    data_container.filterByColumns(full_list)

    if not conf.train_idx:
        train_start, train_end = 241, 408
    else:
        train_start, train_end = conf.train_idx[0], conf.train_idx[1]
        if not (1 <= train_start <= train_end <= 600):
            print('The given start_idx is not correct, use default setting (241 -> 408) instead')
            train_start, train_end = 241, 408

    if not conf.val_idx:
        val_start ,val_end = 409, 420
    else:
        val_start, val_end = conf.val_idx[0], conf.val_idx[1]
        if not (train_end <= val_start <= val_end <= 600):
            print('The given val_idx is not correct, use default setting (409 -> 420) instead')
            val_start, val_end = 409, 420

    if not conf.test_idx:
        test_start, test_end = 421, 434
    else:
        test_start, test_end = conf.test_idx[0], conf.test_idx[1]
        if not (train_end <= test_start <= test_end <= 600):
            print('The given test_idx is not correct, use default setting (421 -> 434) instead')
            test_start, test_end = 421, 434

    train_start, train_end, test_start, test_end = createSplit([train_start, train_end], 
            [val_start, val_end], [test_start, test_end], mode=conf.split_mode)

    assert train_end + 1 == test_start, f'The given series is not continuous {train_end} and {test_start + 1}'

    data_container.filterByIndex(train_start, test_end)
    # TODO: Shift the dataset by hops
    d3m_folder = os.path.join(conf.d3m_root, conf.data_name)
    buildD3MRoot(d3m_folder)
    train_split_df, test_split_df = data_container.generateCSV(d3m_folder, train_start, train_end, test_start, test_end)

    # Finish remaining d3m file generation
    problem_doc = createDummyProblemDoc(conf.data_name, [(y_list[0], len(x_list) + 1)], ['regression'], 'meanSquaredError')
    with open(os.path.join(d3m_folder, 'TRAIN/problem_TRAIN/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    with open(os.path.join(d3m_folder, 'TEST/problem_TEST/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    with open(os.path.join(d3m_folder, 'SCORE/problem_SCORE/problemDoc.json'), 'w') as outfile:
        json.dump(problem_doc, outfile)
    
    tmp_list = train_split_df.columns.tolist()
    tmp_list = [item for item in tmp_list if item not in y_list]
    train_dataset_doc = createDatasetDoc(conf.data_name, 'TRAIN', tmp_list, y_list)
    with open(os.path.join(d3m_folder, 'TRAIN/dataset_TRAIN/datasetDoc.json'), 'w') as outfile:
        json.dump(train_dataset_doc, outfile)
    test_dataset_doc = createDatasetDoc(conf.data_name, 'TEST', tmp_list, y_list)
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

    # Parameters for parquet reading 
    parser.add_argument('--parquet', type=str, required=True, help='Path of the parquet file')
    parser.add_argument('--mode', type=str, default='pgd', help='The mode we will use for parsing: (pgd|cm)')
    parser.add_argument('--engine', type=str, default='pyarrow', help='The engine that pandas used for parquet reading: (pyarrow|fastparquet)')
    parser.add_argument('--train_idx', type=int, nargs=2, help='The start month idx of training split')
    parser.add_argument('--val_idx', type=int, nargs=2, help='The start month idx of validation split')
    parser.add_argument('--test_idx', type=int, nargs=2, help='The start month idx of test split')
    parser.add_argument('--d3m_root', type=str, default='/home/yulin/Desktop/views_dataset', help='The path of output d3m dump')
    parser.add_argument('--data_name', type=str, default='SparkingTest')
    parser.add_argument('--split_mode', type=str, default='train')

    parameters = parser.parse_args()
    main(parameters)

