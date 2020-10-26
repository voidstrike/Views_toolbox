import pandas as pd
import os

from typing import List
from container.BasicContainer import Container
from auxiliary.utils import ConfigurationReader, filterByGrid


# Class for CSV Dataframe
class CSVContainer(Container):
    def __init__(self, root: str, **kwargs) -> None:
        super(CSVContainer, self).__init__(root)
        self.data_type = 'csv/dataframe'

    def _read_data(self, **kwargs) -> None:
        self.data = pd.read_csv(self.root)


# View-specific container
class ViewContainer(CSVContainer):
    def __init__(self, root: str, configuration: ConfigurationReader) -> None:
        super(CSVContainer, self).__init__(root)
        self.conf = configuration
        self.secondary_key = 'pg_id' if self.conf.task_type == 'pgm' else 'country_id'
        self.manipulated = False
        self._read_data()
        self.data_type = 'views/dataframe'
        self.df_idx = pd.IndexSlice  # -> Crop month_idx

    def filterByColumns_(self, feature_list: List[str]) -> pd.DataFrame:
        self.data = self.filterByColumns(feature_list)
        return self.data

    def filterByColumns(self, feature_list: List[str]) -> pd.DataFrame:
        return self.data.filter(items=feature_list)

    def filterByIndex_(self, start: int, end: int) -> pd.DataFrame:
        self.data = self.filterByIndex(start, end)
        return self.data

    def filterByIndex(self, start: int, end: int) -> pd.DataFrame:
        # Hard-code, seems OK here
        # return self.data.loc[self.df_idx[start:end]]
        return self.data.loc[start: end]

    def reorder_index(self, new_order: List[str], sort=True) -> pd.DataFrame:
        tmp = self.data.reorder_levels(new_order)
        return tmp if not sort else tmp.sort_index()

    def _read_data(self, ) -> None:
        # Kind of hardcoded
        tmp_data = pd.read_csv(self.root)
        if self.conf.task_type == 'pgm':
            if self.conf.dv_file_path:
                # Concatenate auxiliary dataset -- DV_DATASET
                print(f'Using auxiliary dataset, MAKE SURE corresponding feature(s) are in the feature list. '
                      f'Current target feature(s): {self.conf.label_list}')
                print('Filtering MAIN DATAFRAME using DV dataset')
                aux_dv_data = pd.read_csv(self.conf.dv_file_path)
                aux_dv_data = aux_dv_data.rename(columns={f'date_{self.conf.shift}': 'month_id'})
                # aux_dv_data['month_id'] = aux_dv_data['month_id'] - 1  # Final Submission
                tmp_data = pd.merge(tmp_data, aux_dv_data, on=['month_id', 'pg_id'])
                tmp_data = tmp_data.drop(columns=['month_id']).rename(columns={'date_t': 'month_id'})
            # Postpone this part to hard_manipulation ?
            if self.conf.filter_file_path:
                print('Further filtering MAIN DATAFRAME using grid dataset')
                aux_grid_data = pd.read_csv(self.conf.filter_file_path)
                tmp_data = filterByGrid(tmp_data, aux_grid_data, mode=self.conf.filter_mode)
                print(len(tmp_data.index))
            self.data = tmp_data.set_index(['month_id', 'pg_id']).sort_index()
        elif self.conf.task_type == 'cm':
            if self.conf.dv_file_path:
                # Concatenate auxiliary dataset -- DV_DATASET
                print(f'Using auxiliary dataset, MAKE SURE corresponding feature(s) are in the feature list. '
                      f'Current target feature(s): {self.conf.label_list}')
                print('Filtering MAIN DATAFRAME using DV dataset')
                aux_dv_data = pd.read_csv(self.conf.dv_file_path)
                aux_dv_data = aux_dv_data.rename(columns={f'date_{self.conf.shift}': 'month_id'})
                # aux_dv_data['month_id'] = aux_dv_data['month_id'] - 1  # Final Submission
                tmp_data = pd.merge(tmp_data, aux_dv_data, on=['month_id', 'country_id'])
                tmp_data = tmp_data.drop(columns=['month_id']).rename(columns={'date_t': 'month_id'})
            self.data = tmp_data.set_index(['month_id', 'country_id']).sort_index()
        else:
            raise RuntimeError(f'Unsupported tasktype : {self.conf.task_type} is given')

    # Predefined operation for the dataframe
    def hard_manipulate(self, csv_test=False):
        # Reorder self.data based on the secondary key
        if not csv_test:
            print('Seems no effect')
        else:
            self.data = pd.get_dummies(self.data, columns=['month', 'country_name'])
        self.manipulated = True

    def to_d3m(self, d3m_root, train_start, train_end, test_start, test_end):
        assert self.manipulated
        # train_split = self.data.loc[self.df_idx[train_start:train_end]]
        train_split = self.data.loc[train_start:train_end]
        train_split = train_split.dropna()
        # test_split = self.data.loc[self.df_idx[test_start:test_end]]
        test_split = self.data.loc[test_start:test_end]
        train_len = len(train_split.index)

        # Noted that index is not drop for now
        tmp_df = pd.concat([train_split, test_split], axis=0).reset_index(drop=False)
        tmp_df.index.name = 'd3mIndex'

        # Save MATCH information for future use
        match_df = tmp_df.filter(items=['month_id', self.secondary_key])
        # match_df = tmp_df.filter(items=['month_id', self.secondary_key])
        match_file_path = os.path.join(d3m_root, 'matching.csv')
        match_df.to_csv(match_file_path)
        tmp_df = tmp_df.drop(columns=['month_id', self.secondary_key])
        # tmp_df = tmp_df.drop(columns=['month_id', self.secondary_key])
        train_split, test_split = tmp_df.loc[:train_len], tmp_df[train_len:]

        # SAVING
        training_file_path = os.path.join(d3m_root, 'TRAIN/dataset_TRAIN/tables/learningData.csv')
        train_split.to_csv(training_file_path)
        test_file_path = os.path.join(d3m_root, 'TEST/dataset_TEST/tables/learningData.csv')
        test_split.to_csv(test_file_path)
        test_file_path = os.path.join(d3m_root, 'SCORE/dataset_SCORE/tables/learningData.csv')
        test_split.to_csv(test_file_path)

        return train_split, test_split
