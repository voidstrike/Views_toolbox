import os
import json
import pandas as pd

_UNKNOWN_TYPE = 'UNKNOWN_TYPE'


# Auxiliary object to read views_configuration
class ConfigurationReader(object):
    def __init__(self, root: str) -> None:
        self.root = root
        if not os.path.exists(self.root):
            raise RuntimeError(f'Configuration file at {self.root} does not exist')

        self._read_configuration()
        pass

    def _read_configuration(self, ) -> None:
        with open(self.root, 'r') as f:
            setting_corpus = json.load(f)

        # Basic configuration -- Mandatory
        self.task_type = setting_corpus.get('task_type', _UNKNOWN_TYPE)
        self.file_type = setting_corpus.get('file_type', None)
        self.file_path = setting_corpus.get('file_path', './')
        self.dv_file_path = setting_corpus.get('dv_file_path', None)
        self.filter_file_path = setting_corpus.get('filter_file_path', None)
        self.filter_mode = setting_corpus.get('filter_mode', 1)
        self.engine = setting_corpus.get('engine', None)
        self.shift = setting_corpus.get('shift', 1)

        # Feature list
        self.feature_list = setting_corpus['features'] if 'features' in setting_corpus else []
        self.label_list = setting_corpus['labels'] if 'labels' in setting_corpus else []

        # TRAIN/VAL/TEST
        self.train_start, self.train_end = setting_corpus.get('train_split', [241, 408])
        self.val_start, self.val_end = setting_corpus.get('val_split', [409, 420])
        self.test_start, self.test_end = setting_corpus.get('test_split', [421, 434])

        # D3M and split mode
        self.d3m_root = setting_corpus.get('d3m_root', './')
        self.data_name = setting_corpus.get('data_name', '0')
        self.mode = setting_corpus.get('mode', 'train')  # or train_val

        if not isinstance(self.feature_list, list):
            self.feature_list = [self.feature_list]
        if not isinstance(self.label_list, list):
            self.label_list = [self.label_list]
        pass


# Filter Function for Hurdle Model
def filterByGrid(in_dataframe: pd.DataFrame, grid_frame: pd.DataFrame, mode: int) -> pd.DataFrame:
    if mode == 1:
        grid_frame = grid_frame[grid_frame['conflict_count'] > 0]
    elif mode == 2:
        grid_frame = grid_frame[grid_frame['conflict_count'] > 1]
    elif mode == 3:
        grid_frame = grid_frame[grid_frame['max_span'] > 1]
    else:
        raise RuntimeError('Unsupported filter mode!!!')
    print(len(grid_frame.index))
    return pd.merge(in_dataframe, grid_frame, on=['pg_id'])
