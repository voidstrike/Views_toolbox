import pandas as pd

from typing import List, Tuple
from container import BasicContainer


# Class for Parquet Dataframe
class ParquetContainer(BasicContainer):
    def __init__(self, root: str, engine: str) -> None:
        super(ParquetContainer, self).__init__(root)
        self.engine = engine
        self.data_type = 'parquet/dataframe'

    def _read_data(self,) -> None:
        self.data = pd.read_parquet(self.root, engine=self.engine)


# View-specific container
class ViewContainer(ParquetContainer):
    def __init__(self, root: str, engine: str) -> None:
        super(ParquetContainer, self).__init__(root, engine)
        self._read_data()
        self.data_type = 'views/dataframe'
        self.df_idx = pd.IndexSlice

    def filterByColumns_(self, feature_list: List[str]) -> pd.DataFrame:
        self.data = self.filterByColumns(feature_list)
        return self.data

    def filterByColumns(self, feature_list: List[str]) -> pd.DataFrame:
        return self.data.filter(items=feature_list)

    def filterByIndex_(self, start: int ,end: int) -> pd.DataFrame:
        self.data = self.filterByIndex(start, end)
        return self.data

    def filterByIndex(self, start: int, end: int) -> pd.DataFrame:
        # Hard-code, seems OK here
        return self.data.loc[self.df_idx[start:end]]

    def reorder_index(self, new_order: List[str], sort=True) ->pd.DataFrame:
        tmp = self.data.reorder_levels(new_order)
        return tmp if not sort else tmp.sort_index()

    # Predefined operation for the dataframe
